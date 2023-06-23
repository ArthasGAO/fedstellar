# 
# This file is part of the fedstellar framework (see https://github.com/enriquetomasmb/fedstellar).
# Copyright (c) 2022 Enrique Tomás Martínez Beltrán.
# 


import logging
import math
import pickle
import copy
import time
import wandb

import torch
from statistics import mean

from typing import List, Dict, OrderedDict, Optional, TypedDict

from pytorch_lightning.loggers import wandb

from fedstellar.learning.aggregators.aggregator import Aggregator
from fedstellar.learning.aggregators.helper import cosine_similarity
from fedstellar.learning.modelmetrics import ModelMetrics
from fedstellar.learning.pytorch.lightninglearner import LightningLearner
from fedstellar.learning.aggregators.helper import normalise_layers

MIN_MAPPED_LOSS = float(0.01)
COSINE_FILTER_THRESHOLD = float(0.5)


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


# hoooold the dooor
def map_loss(loss) -> float:
    mapped_loss = math.exp(-loss)
    if (math.isnan(mapped_loss)) | (mapped_loss < MIN_MAPPED_LOSS):
        # logging.error("[Sentinel] Mapped loss is invalid! Return small value instead")
        return float(MIN_MAPPED_LOSS)
    return mapped_loss


# maps loss of neighbours relative distance to reference model
# inverse own loss serves as damping factors to allow more aggregation in early FL rounds
def map_loss_distance(loss: float, my_loss: float) -> float:
    loss_dist = loss - my_loss
    # prevent division by zero
    if my_loss <= MIN_MAPPED_LOSS:
        my_loss = MIN_MAPPED_LOSS
    k = 1 / my_loss
    mapped_distance_loss = math.exp(-k * loss_dist)
    if (math.isnan(mapped_distance_loss)) | (mapped_distance_loss < MIN_MAPPED_LOSS):
        # logging.error("[Sentinel] Mapped loss is invalid! Return small value instead")
        return float(0)
    return mapped_distance_loss


# boy gets flipped by label-flip
def map_loss2(loss) -> float:
    return 1 / (abs(loss) + 1)


class SentinelGlobal(Aggregator):
    """
    Sentinel
        Custom aggregation method based on cosine similarity, loss distance and normalisation.
    """

    def __init__(self, node_name="unknown", config=None, logger=None, learner=None, agg_round=0,
                 global_trust: Dict[int, Dict[str, Dict[str, int]]] = None):
        super().__init__(node_name, config, logger, learner, agg_round, global_trust)
        self.config = config
        self.role = self.config.participant["device_args"]["role"]
        logging.info("[Sentinel] My config is {}".format(self.config))
        self.logger = logger
        self.learner: LightningLearner = learner
        self.agg_round = agg_round
        # global_trust holds trust scores per round for each neighbour
        # example:
        #   {0: {'node1': {'node2': 0, 'node3': 1}, 'node2': {'node1': 1, 'node3': 0}},
        #   {1: {'node1': {'node2': 0, 'node3': 0}, 'node2': {'node1': 0, 'node3': 1}},
        self.global_trust = global_trust
        self.global_trust[agg_round] = {}
        self.global_trust[agg_round][self.node_name] = {}
        logging.info("[Sentinel] My global trust is {}".format(self.global_trust))

    def add_model(self, model: OrderedDict, nodes: List[str], metrics: ModelMetrics):
        logging.info("[Sentinel.add_model] Computing metrics for node(s): {}".format(nodes))
        logging.info("[Sentinel.add_model] Received model at round {} with trust scores {}".
                     format(self.agg_round, metrics.global_trust))

        # the key should always be the previous round, just for safety
        # TODO: need a different way to add these under the same round
        for node in nodes:
            for key in metrics.global_trust.keys():
                self.global_trust[key][node] = metrics.global_trust[key]

        logging.info("[Sentinel.add_model] New trust scores: {}".format(self.global_trust))

        model_params = model
        tmp_model = copy.deepcopy(self.learner.latest_model)
        tmp_model.load_state_dict(model_params)
        val_loss, val_acc = self.learner.validate_neighbour_no_pl2(tmp_model)
        tmp_model = None
        # -> with validate_neighbour_pl:
        # ReferenceError: weakly-referenced object no longer exists (at local_params = self.learner.get_parameters())

        if nodes is not None:
            for node in nodes:
                if node != self.node_name:
                    mapping = {f'val_loss_{node}': val_loss}
                    self.learner.logger.log_metrics(metrics=mapping, step=self.learner.logger.global_step)

        local_params = self.learner.get_parameters()
        cos_similarity: float = cosine_similarity(local_params, model_params)

        if nodes is not None:
            for node in nodes:
                if node != self.node_name:
                    mapping = {f'cos_{node}': cos_similarity}
                    self.learner.logger.log_metrics(metrics=mapping, step=self.learner.logger.global_step)

        metrics.cosine_similarity = cos_similarity
        metrics.validation_loss = val_loss
        metrics.validation_accuracy = val_acc

        logging.info("[Sentinel.add_model] Computed metrics for node(s): {}: --> {}".format(nodes, metrics))

        super().add_model(model=model, nodes=nodes, metrics=metrics)

    def aggregate(self, models):

        logging.info("[Sentinel]: Aggregation round {}".format(self.agg_round))

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
        for node, msg in filtered_models.items():
            params = msg[0]
            metrics: ModelMetrics = msg[1]
            loss[node] = metrics.validation_loss
            mapped_loss[node] = map_loss_distance(metrics.validation_loss, my_loss)
            cos[node] = metrics.cosine_similarity
        malicous_by_loss = {key for key, loss in mapped_loss.items() if loss == 0}

        logging.info("[Sentinel]: Loss metrics: {}".format(loss))
        logging.info("[Sentinel]: Loss mapped metrics: {}".format(mapped_loss))
        logging.info("[Sentinel]: Cos metrics: {}".format(cos))

        # Step 3: Normalise the untrusted models
        untrusted_models = {k: filtered_models[k] for k in filtered_models.keys() - {self.node_name}}
        normalised_models = {}
        for key in untrusted_models.keys():
            normalised_models[key] = normalise_layers(untrusted_models[key], my_model)
        normalised_models[self.node_name] = my_model

        # Create a Zero Model
        accum = (list(filtered_models.values())[-1][0]).copy()
        for layer in accum:
            accum[layer] = torch.zeros_like(accum[layer])

        # Aggregate
        total_mapped_loss: float = sum(mapped_loss.values())
        logging.info("Sentinel: Total mapped loss: {}".format(total_mapped_loss))
        for node, message in filtered_models.items():
            client_model = message[0]
            for layer in client_model:
                accum[layer] = accum[layer] + client_model[layer] * mapped_loss[node]
                mapping = {f'agg_weight_{node}': mapped_loss[node] / total_mapped_loss}
                self.learner.logger.log_metrics(metrics=mapping, step=0)

        # Normalize accumulated model wrt loss
        for layer in accum:
            accum[layer] = accum[layer] / total_mapped_loss

        malicious = malicous_by_cosine.union(malicous_by_loss)
        logging.info("Sentinel: Tagged as malicious: {}".format(list(malicious)))

        # Store global trust scores
        logging.info("Sentinel: Global trust scores before adding: {}".format(self.global_trust))
        for node_key in models.keys():
            if node_key in malicious:
                self.global_trust[self.agg_round][self.node_name].update({node_key: 0})
            else:
                self.global_trust[self.agg_round][self.node_name].update({node_key: 1})

        logging.info("Sentinel: Global trust scores after adding: {}".format(self.global_trust))

        return accum
