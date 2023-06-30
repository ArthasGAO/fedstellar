# 
# This file is part of the fedstellar framework (see https://github.com/enriquetomasmb/fedstellar).
# Copyright (c) 2022 Enrique Tomás Martínez Beltrán.
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
        if metrics.cosine_similarity > threshold:
            filtered[node] = msg
        else:
            logging.info("[Sentinel] Filtering model due to low cos similarity {}".format(node))
    return filtered


"""
# maps loss of neighbours relative distance to reference model
# inverse own loss serves as damping factors to allow more aggregation in early FL rounds
def map_loss_distance(loss: float, my_loss: float, threshold: float) -> float:
    # prevent division by zero
    k = 1 / max(MIN_LOSS, my_loss)
    loss_dist = max(loss - my_loss, 0)
    mapped_distance_loss = math.exp(-k * loss_dist)
    if (mapped_distance_loss < threshold) | (math.isnan(mapped_distance_loss)):
        return float(0)
    return mapped_distance_loss
"""


class Sentinel(Aggregator):
    """
    Sentinel
        Custom aggregation method based on cosine similarity, loss distance and normalisation.
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

        logging.info("[Sentinel] My config is {}".format(self.config))
        logging.info("[Sentinel] My loss distance threshold is {}".format(self.loss_dist_threshold))
        logging.info("[Sentinel] My current loss history is {}:".format(self.loss_history))

    def add_model(self, model: OrderedDict, nodes: List[str], metrics: ModelMetrics):

        logging.info("[Sentinel.add_model] Computing metrics for node(s): {}".format(nodes))

        model_params = model

        tmp_model = copy.deepcopy(self.learner.latest_model)
        tmp_model.load_state_dict(model_params)
        val_loss, val_acc = self.learner.validate_neighbour_no_pl2(tmp_model)
        tmp_model = None
        # -> with validate_neighbour_pl:
        # ReferenceError: weakly-referenced object no longer exists (at local_params = self.learner.get_parameters())

        if nodes is not None:
            for node in nodes:
                mapping = {f'val_loss_{node}': val_loss}
                self.learner.logger.log_metrics(metrics=mapping, step=self.learner.logger.global_step)

        local_params = self.learner.get_parameters()
        cos_similarity: float = cosine_similarity(local_params, model_params)

        if nodes is not None:
            for node in nodes:
                mapping = {f'cos_{node}': cos_similarity}
                self.learner.logger.log_metrics(metrics=mapping, step=self.learner.logger.global_step)

        metrics.cosine_similarity = cos_similarity
        metrics.validation_loss = val_loss
        metrics.validation_accuracy = val_acc

        logging.info("[Sentinel.add_model] Computed metrics for node(s): {}: --> {}".format(nodes, metrics))

        super().add_model(model=model, nodes=nodes, metrics=metrics)

    def get_mapped_avg_loss(self, node_key: str, loss: float, my_loss: float, threshold: float) -> float:
        # calculate next average loss
        prev_loss_hist: list = self.loss_history.get(node_key, [])
        prev_loss_hist.append(loss)
        self.loss_history[node_key] = prev_loss_hist
        avg_loss = mean(prev_loss_hist)
        mapping = {f'avg_loss_{node_key}': avg_loss}
        self.learner.logger.log_metrics(metrics=mapping, step=self.learner.logger.global_step)

        # fallback to current loss
        local_loss_hist = self.loss_history.get(self.node_name, [loss])
        avg_local_loss = mean(local_loss_hist)

        k = 1 / max(MIN_LOSS, avg_local_loss)
        loss_dist = max(avg_loss - avg_local_loss, 0)
        mapped_distance_loss = math.exp(-k * loss_dist)
        if (mapped_distance_loss < threshold) | (math.isnan(mapped_distance_loss)):
            return float(0)
        return mapped_distance_loss

    def aggregate(self, models):

        # logging.info("Current logger step: {}".format(self.logger.global_step))
        # logging.info("Current learner step: {}".format(self.learner.logger.global_step))

        """
         Krum selects one of the m local models that is similar to other models
         as the global model, the Euclidean distance between two local models is used.

         Args:
             models: Dictionary with the models (node: model,num_samples).
         """
        # Check if there are models to aggregate

        if len(models) == 0:
            logging.warning("[Sentinel] Trying to aggregate models when there is no models")
            return None

        # The model of the aggregator serves as a trusted reference
        my_model = models.get(self.node_name)  # change
        if my_model is None:
            logging.warning("[Sentinel] Trying to aggregate models when bootstrap is not available")
            return None

        # Step 1: Evaluate cosine similarity
        filtered_models = filter_models_by_cosine(models, COSINE_FILTER_THRESHOLD)
        malicous_by_cosine = models.keys() - filtered_models.keys()
        if len(filtered_models) == 0:
            logging.warning("Sentinel: No more models to aggregate after filtering!")
            return None
        for node_key in malicous_by_cosine:
            mapping = {f'agg_weight_{node_key}': 0}
            self.learner.logger.log_metrics(metrics=mapping, step=self.learner.logger.global_step)

        # Step 2: Evaluate validation (bootstrap) loss
        my_loss = my_model[1].validation_loss
        loss: Dict = {}
        mapped_loss: Dict = {}
        cos: Dict = {}
        for node_key, msg in filtered_models.items():
            params = msg[0]
            metrics: ModelMetrics = msg[1]
            loss[node_key] = metrics.validation_loss
            mapped_loss[node_key] = self.get_mapped_avg_loss(node_key, metrics.validation_loss, my_loss, self.loss_dist_threshold)
            cos[node_key] = metrics.cosine_similarity
        malicous_by_loss = {key for key, loss in mapped_loss.items() if loss == 0}

        logging.info("[Sentinel]: Loss metrics: {}".format(loss))
        logging.info("[Sentinel]: Loss mapped metrics: {}".format(mapped_loss))
        logging.info("[Sentinel]: Cos metrics: {}".format(cos))

        # Step 3: Normalise the remaining (filtered) untrusted models
        untrusted_models = {k: filtered_models[k] for k in filtered_models.keys() - {self.node_name}}
        normalised_models = {}
        for key in untrusted_models.keys():
            normalised_models[key] = normalise_layers(untrusted_models[key], my_model)
        normalised_models[self.node_name] = my_model

        # Create a Zero Model
        accum = (list(normalised_models.values())[-1][0]).copy()
        for layer in accum:
            accum[layer] = torch.zeros_like(accum[layer])

        # Aggregate
        total_mapped_loss: float = sum(mapped_loss.values())
        logging.info("Sentinel: Total mapped loss: {}".format(total_mapped_loss))
        for node, message in normalised_models.items():
            client_model = message[0]
            for layer in client_model:
                accum[layer] = accum[layer] + client_model[layer] * mapped_loss[node]
                mapping = {f'agg_weight_{node}': mapped_loss[node] / total_mapped_loss,
                           f'mapped_loss_{node}': mapped_loss[node]}
                self.learner.logger.log_metrics(metrics=mapping, step=self.learner.logger.global_step)

        # Normalize accumulated model wrt loss
        for layer in accum:
            accum[layer] = accum[layer] / total_mapped_loss

        malicious = malicous_by_cosine.union(malicous_by_loss)
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
