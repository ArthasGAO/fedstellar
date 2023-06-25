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
from fedstellar.learning.aggregators.sentinel import filter_models_by_cosine
from fedstellar.learning.aggregators.sentinel import map_loss_distance

MIN_MAPPED_LOSS = float(0.01)
COSINE_FILTER_THRESHOLD = float(0.5)


class SentinelGlobal(Aggregator):
    """
    SentinelGobal
        Based on Sentinel for local trust.
        Additionally, establishes global trust by relying on trust neighbour opinions.
    """

    def __init__(self, node_name="unknown", config=None, logger=None, learner=None, agg_round=0,
                 global_trust: OrderedDict[int, Dict[str, Dict[str, int]]] = None, active_round=3):
        super().__init__(node_name, config, logger, learner, agg_round)
        self.config = config
        self.role = self.config.participant["device_args"]["role"]
        logging.info("[SentinelGlobal] My config is {}".format(self.config))
        logging.info("[SentinelGlobal] Configured to round {}".format(self.agg_round))
        self.logger = logger
        self.learner: LightningLearner = learner
        # global_trust holds trust scores for each round and neighbour
        # example:
        #   {0: {'node1': {'node2': 0, 'node3': 1}, 'node2': {'node1': 1, 'node3': 0}},
        #   {1: {'node1': {'node2': 0, 'node3': 0}, 'node2': {'node1': 0, 'node3': 1}},
        #  -> at round 0, node1 identified node2 as malicious (0), node3 as trusted (1).
        self.global_trust = global_trust
        self.global_trust[self.agg_round] = {}
        self.global_trust[self.agg_round][self.node_name] = {}
        logging.info("[SentinelGlobal] My global trust is {}".format(self.global_trust))

    def broadcast_local_model(self):
        logging.info("[SentinelGlobal.broadcast_local_model]. Partial aggregation: only local model")
        logging.info("[SentinelGlobal.broadcast_local_model]. Sending local global trust: {}".format(self.global_trust))
        for node, (model, metrics) in list(self._models.items()):
            current_global_trust = copy.deepcopy(self.global_trust)
            if node == self.node_name:
                return model, [node], ModelMetrics(
                    num_samples=metrics.num_samples,
                    global_trust=current_global_trust)
        return None

    def get_trusted_neighbour_opinion(self, target_node: str) -> float:
        try:
            # Caveat: round 0 is just a diffusion round, thus agg_round - 2
            prev_round = self.agg_round - 2
            prev_global_trust = self.global_trust[prev_round]
            prev_local_trust = prev_global_trust[self.node_name]
            trusted_neighbours = []
            # Collect trusted neighbours
            for node_key in prev_local_trust.keys():
                if prev_local_trust[node_key] == 1:
                    trusted_neighbours.append(node_key)
            # Collect and average trusted neighbour opinion
            sum_trust = 0
            for trusted_nei in trusted_neighbours:
                nei_trust = prev_global_trust[trusted_nei][target_node]
                sum_trust += nei_trust
            avg_trust = sum_trust / len(trusted_neighbours)
            logging.info("[SentinelGlobal.get_trusted_neighbour_opinion] Avg. trusted neighbour opinion for node {}: {}"
                         .format(target_node, avg_trust))
            mapping = {f'Global Trust {target_node}': avg_trust}
            self.learner.logger.log_metrics(metrics=mapping, step=self.learner.logger.global_step)
        except KeyError:
            logging.error("[SentinelGlobal.get_trusted_neighbour_opinion] KeyError for round {}, self.node_name: {}, target_node: {}"
                          .format(self.agg_round, self.node_name, target_node))
            logging.error("[SentinelGlobal.get_trusted_neighbour_opinion] With global trust: {}"
                          .format(self.global_trust))
            avg_trust = 0
        return avg_trust


    def add_model(self, model: OrderedDict, nodes: List[str], metrics: ModelMetrics):
        logging.info("[SentinelGlobal.add_model] Computing metrics for node(s): {}".format(nodes))

        #logging.info("[SentinelGlobal.add_model] Received model at round {} with trust scores {}".format(self.agg_round, metrics.global_trust))

        # TODO: maybe only share last round with model?
        neighbor_opinions = metrics.global_trust
        for round_key in neighbor_opinions.keys():
            for node_key in neighbor_opinions[round_key]:
                # local trust is added during aggregation
                if node_key != self.node_name:
                    self.global_trust[round_key][node_key] = neighbor_opinions[round_key][node_key]

        # logging.info("[SentinelGlobal.add_model] New trust scores: {}".format(self.global_trust))

        if self.agg_round > 2 and nodes is not None:
            for node_key in nodes:
                avg_global_trust = self.get_trusted_neighbour_opinion(node_key)
                logging.info("[SentinelGlobal.add_model] node(s) {} since not trusted".format(nodes, metrics))
                if avg_global_trust < 0.5:
                    metrics.cosine_similarity = 0
                    metrics.validation_loss = float('inf')
                    metrics.validation_accuracy = 0
                    super().add_model(model=model, nodes=nodes, metrics=metrics)
                    return

        # TODO probably push down evaluation down to aggregation, since not all trust scores are available here
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

        logging.info("[SentinelGlobal.add_model] Computed metrics for node(s): {}: --> {}".format(nodes, metrics))

        super().add_model(model=model, nodes=nodes, metrics=metrics)

    def clear(self):
        """
        Clear all for a new aggregation.
        """
        observers = self.get_observers()
        next_round = self.agg_round + 1
        prev_global_trust = copy.deepcopy(self.global_trust)
        self.__init__(node_name=self.node_name,
                      config=self.config,
                      logger=self.logger,
                      learner=self.learner,
                      agg_round=next_round,
                      global_trust=prev_global_trust)
        for o in observers:
            self.add_observer(o)

    def aggregate(self, models):

        logging.info("[SentinelGlobal]: Aggregation round {}".format(self.agg_round))

        """
         Krum selects one of the m local models that is similar to other models
         as the global model, the Euclidean distance between two local models is used.

         Args:
             models: Dictionary with the models (node: model,num_samples).
         """
        # Check if there are models to aggregate

        if len(models) == 0:
            logging.warning("[SentinelGlobal] Trying to aggregate models when there is no models")
            return None

        # The model of the aggregator serves as a trusted reference
        my_model = models.get(self.node_name)  # change
        if my_model is None:
            logging.warning("[SentinelGlobal] Trying to aggregate models when bootstrap is not available")
            return None

        # Step 1: Evaluate cosine similarity
        filtered_models = filter_models_by_cosine(models, COSINE_FILTER_THRESHOLD)
        malicous_by_cosine = models.keys() - filtered_models.keys()
        if len(filtered_models) == 0:
            logging.warning("[SentinelGlobal]: No more models to aggregate after filtering!")
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

        logging.info("[SentinelGlobal]: Loss metrics: {}".format(loss))
        logging.info("[SentinelGlobal]: Loss mapped metrics: {}".format(mapped_loss))
        logging.info("[SentinelGlobal]: Cos metrics: {}".format(cos))

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
        logging.info("[SentinelGlobal]: Total mapped loss: {}".format(total_mapped_loss))
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
        logging.info("[SentinelGlobal]: Tagged as malicious: {}".format(list(malicious)))

        # Store global trust scores
        for node_key in models.keys():
            # The aggregator has already proceeded to the next round, thus
            if node_key in malicious:
                self.global_trust[self.agg_round][self.node_name][node_key] = 0
            else:
                self.global_trust[self.agg_round][self.node_name][node_key] = 1
        # logging.info("[SentinelGlobal]: Global trust scores after adding: {}".format(self.global_trust))

        return accum
