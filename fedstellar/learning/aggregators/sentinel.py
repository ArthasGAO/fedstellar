# 
# This file is part of the Fedstellar platform (see https://github.com/enriquetomasmb/fedstellar).
# Copyright (c) 2023 Janosch Baltensperger.
#
from fedstellar.learning.aggregators.aggregator import Aggregator
from fedstellar.learning.aggregators.helper import cosine_similarity
from fedstellar.learning.modelmetrics import ModelMetrics
from fedstellar.learning.pytorch.lightninglearner import LightningLearner
from fedstellar.learning.aggregators.helper import normalise_layers
from statistics import mean
from typing import List, Dict, OrderedDict, Set
import logging
import math
import copy
import torch

COSINE_FILTER_THRESHOLD = float(0.5)
MIN_LOSS = float(0.001)


def filter_models_by_cosine(models: Dict, threshold: float) -> Dict:
    filtered: Dict = {}
    for node, msg in models.items():
        params = msg[0]
        metrics: ModelMetrics = msg[1]
        if metrics.cosine_similarity < threshold:
            logging.info("[Sentinel] Filtering model due to low cos similarity {}".format(node))
        else:
            filtered[node] = msg
    return filtered


class Sentinel(Aggregator):
    """
    Sentinel
        Custom aggregation method based on model similarity, bootstrap validation and normalisation.
    """

    def __init__(self, node_name="unknown",
                 config=None,
                 logger=None,
                 learner=None,
                 agg_round=0,
                 loss_distance_threshold=0.1,
                 loss_history=None
                 ):
        super().__init__(node_name, config, logger, learner, agg_round)
        self.config = config
        self.role = self.config.participant["device_args"]["role"]
        self.logger = logger
        self.learner: LightningLearner = learner
        self.loss_dist_threshold = loss_distance_threshold
        self.loss_history: Dict[str, list] = loss_history or {}
        self.similarity_threshold = COSINE_FILTER_THRESHOLD

        logging.info("[Sentinel] My config is {}".format(self.config))
        logging.info("[Sentinel] My loss distance threshold is {}".format(self.loss_dist_threshold))
        logging.info("[Sentinel] My current loss history is {}:".format(self.loss_history))

    def add_model(self, model: OrderedDict, nodes: List[str], metrics: ModelMetrics):
        super().add_model(model=model, nodes=nodes, metrics=metrics)

    def evaluate_model(self, model: OrderedDict, node: str, metrics: ModelMetrics):

        # Cosine Similarity
        model_params = model
        local_params = self.learner.get_parameters()
        cos_similarity: float = cosine_similarity(local_params, model_params)

        # Loss
        if cos_similarity < COSINE_FILTER_THRESHOLD:
            val_loss = float('inf')
            val_acc = float(0)
        else:
            tmp_model = copy.deepcopy(self.learner.latest_model)
            tmp_model.load_state_dict(model_params)
            val_loss, val_acc = self.learner.validate_neighbour_model(tmp_model)
            tmp_model = None

        # Log model metrics
        if node != self.node_name:
            mapping = {f'val_loss_{node}': val_loss, f'cos_{node}': cos_similarity}
            self.learner.logger.log_metrics(metrics=mapping, step=0)

        metrics.cosine_similarity = cos_similarity
        metrics.validation_loss = val_loss
        metrics.validation_accuracy = val_acc

        return metrics

    def get_mapped_avg_loss(self, node_key: str, loss: float) -> float:
        # calculate next average loss
        prev_loss_hist: list = self.loss_history.get(node_key, [])
        prev_loss_hist.append(loss)
        self.loss_history[node_key] = prev_loss_hist  # update loss history
        avg_loss = mean(prev_loss_hist)
        mapping = {f'avg_loss_{node_key}': avg_loss}
        self.learner.logger.log_metrics(metrics=mapping, step=0)

        # don't consider neighbours in round 0, since diffusion (FEDSTELLAR specific)
        if self.agg_round == 0 and node_key != self.node_name:
            return float(0)

        # fallback to current loss
        local_loss_hist = self.loss_history.get(self.node_name, [loss])
        avg_local_loss = mean(local_loss_hist)

        k = 1 / max(MIN_LOSS, avg_local_loss)
        loss_dist = max(avg_loss - avg_local_loss, 0)
        mapped_distance_loss = math.exp(-k * loss_dist)
        if (mapped_distance_loss < self.loss_dist_threshold) | (math.isnan(mapped_distance_loss)):
            return float(0)
        return mapped_distance_loss

    def aggregate(self, models):

        logging.info("[Sentinel]: Aggregation round {}".format(self.agg_round))

        if len(models) == 0:
            logging.warning("[Sentinel] Trying to aggregate models when there is no models")
            return None

        # Compute metrics
        for node_key in models.keys():
            model = models[node_key][0]
            metrics: ModelMetrics = models[node_key][1]
            # the own local model also requires eval to get loss distance
            metrics_eval = self.evaluate_model(model, node_key, metrics)
            models[node_key] = (model, metrics_eval)

        # The model of the aggregator serves as a trusted reference
        local_params = models.get(self.node_name)[0]
        if local_params is None:
            logging.warning("[Sentinel] Trying to aggregate models when bootstrap is not available")
            return None

        # Step 1: Evaluate cosine similarity
        filtered_models = filter_models_by_cosine(models, self.similarity_threshold)
        malicious_by_cosine = models.keys() - filtered_models.keys()
        if len(filtered_models) == 0:
            logging.warning("Sentinel: No more models to aggregate after filtering!")
            return models.get(self.node_name)[0]

        for node_key in malicious_by_cosine:
            prev_loss_hist: list = self.loss_history.get(node_key, [])
            avg_loss = mean(prev_loss_hist) if prev_loss_hist else -1
            mapping = {f'agg_weight_{node_key}': 0,
                       f'mapped_loss_{node_key}': 0,
                       f'avg_loss_{node_key}': avg_loss}
            self.learner.logger.log_metrics(metrics=mapping, step=0)

        # Step 2: Evaluate validation (bootstrap) loss
        loss: Dict = {}; mapped_loss: Dict = {}; cos: Dict = {}
        for node_key, msg in filtered_models.items():
            metrics: ModelMetrics = msg[1]
            loss[node_key] = metrics.validation_loss
            mapped_loss[node_key] = self.get_mapped_avg_loss(node_key, metrics.validation_loss)
            cos[node_key] = metrics.cosine_similarity
        malicious_by_loss = {key for key, loss in mapped_loss.items() if loss == 0}

        # Step 3: Normalise the remaining (filtered) untrusted models
        normalised_models = {}
        for key, msg in filtered_models.items():
            model_params = msg[0]
            metrics = msg[1]
            if key == self.node_name:
                normalised_models[key] = (local_params, metrics)
            else:
                normalized_params = normalise_layers(model_params, local_params)
                normalised_models[key] = (normalized_params, metrics)

        # Create a Zero Model
        accum = (list(normalised_models.values())[-1][0]).copy()
        for layer in accum:
            accum[layer] = torch.zeros_like(accum[layer])

        # Aggregate
        total_mapped_loss: float = sum(mapped_loss.values())
        logging.info("Sentinel: Total mapped loss: {}".format(total_mapped_loss))
        for node, message in normalised_models.items():
            model = message[0]
            weight = mapped_loss[node] / total_mapped_loss
            for layer in model:
                accum[layer] = accum[layer] + model[layer] * weight

            mapping = {f'agg_weight_{node}': mapped_loss[node] / total_mapped_loss,
                       f'mapped_loss_{node}': mapped_loss[node]}
            self.learner.logger.log_metrics(metrics=mapping, step=0)

        malicious = malicious_by_cosine.union(malicious_by_loss)
        logging.info("Sentinel: Tagged as malicious: {}".format(list(malicious)))

        return accum

    def clear(self):
        """
        Clear all for a new aggregation.
        """
        observers = self.get_observers()
        next_round = self.agg_round + 1
        self.__init__(node_name=self.node_name,
                      config=self.config,
                      logger=self.logger,
                      learner=self.learner,
                      agg_round=next_round,
                      loss_distance_threshold=self.loss_dist_threshold,
                      loss_history=self.loss_history)
        for o in observers:
            self.add_observer(o)
