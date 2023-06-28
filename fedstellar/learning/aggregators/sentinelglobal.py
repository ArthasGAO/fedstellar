# 
# This file is part of the fedstellar framework (see https://github.com/enriquetomasmb/fedstellar).
# Copyright (c) 2022 Enrique Tomás Martínez Beltrán.
#

import logging
import math
import pickle
import copy
import pprint

import torch
import pandas as pd

pd.options.display.max_columns = None
from statistics import mean

from typing import List, Dict, OrderedDict, Optional, TypedDict, Set

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
DEFAULT_NEI_TRUST = 0
TRUST_THRESHOLD = 0.5


class SentinelGlobal(Aggregator):
    """
    SentinelGlobal
        Based on Sentinel for local trust.
        Additionally, establishes global trust by relying on trust neighbour opinions.
    """

    def __init__(self, node_name="unknown",
                 config=None, logger=None,
                 learner=None, agg_round=0,
                 global_trust: OrderedDict[int, Dict[str, Dict[str, int]]] = None,
                 active_round=3,
                 num_evals=0,
                 neighbor_keys=None):
        super().__init__(node_name, config, logger, learner, agg_round)
        self.config = config
        self.role = self.config.participant["device_args"]["role"]
        logging.info("[SentinelGlobal] My config is {}".format(self.config))
        logging.info("[SentinelGlobal] Configured to round {}".format(self.agg_round))
        self.logger = logger
        self.learner: LightningLearner = learner
        self.active_round = active_round
        self.num_evals = num_evals
        # global_trust holds trust scores for each round and neighbour
        # example:
        #   {0: {'node1': {'node2': 0, 'node3': 1}, 'node2': {'node1': 1, 'node3': 0}},
        #   {1: {'node1': {'node2': 0, 'node3': 0}, 'node2': {'node1': 0, 'node3': 1}},
        #  -> at round 0, node1 identified node2 as malicious (0), node3 as trusted (1).
        self.global_trust = global_trust
        self.neighbor_keys: Set = neighbor_keys if neighbor_keys is not None else set()
        logging.info("SentinelGlobal: neighbours set to {}".format(self.neighbor_keys))
        self.global_trust[self.agg_round] = {}
        self.global_trust[self.agg_round][self.node_name] = {}
        # logging.info("[SentinelGlobal] My prev global trust is {}".format(self.global_trust[self.agg_round-2]))

    def broadcast_local_model(self):
        logging.info("[SentinelGlobal.broadcast_local_model]. Partial aggregation: only local model")
        for node, (model, metrics) in list(self._models.items()):
            current_global_trust = copy.deepcopy(self.global_trust)
            """
            for round_key in range(0, self.agg_round - 2):
                for node_key in current_global_trust[round_key]:
                    assert len(current_global_trust[round_key][node_key]) != 0
            """
            if node == self.node_name:
                return model, [node], ModelMetrics(
                    num_samples=metrics.num_samples,
                    global_trust=current_global_trust)
        return None

    def get_trusted_neighbour_opinion(self, target_node: str) -> float:
        # A node always trusts itself, regardless of neighbor opinions
        if target_node == self.node_name:
            avg_trust = 1
            logging.info("[SentinelGlobal.get_trusted_neighbour_opinion] Avg. trusted neighbour opinion for node {}: {}"
                         .format(target_node, avg_trust))
            mapping = {f'Global Trust {target_node}': avg_trust}
            self.learner.logger.log_metrics(metrics=mapping, step=self.learner.logger.global_step)
            return avg_trust

        # Caveat: round 0 is just a diffusion round, thus agg_round - 2
        prev_round = self.agg_round - 2
        prev_global_trust = self.global_trust[prev_round]
        prev_local_trust = prev_global_trust[self.node_name]
        curr_trusted_nei = None
        avg_trust = 0
        try:
            trusted_neighbours = []
            # Collect trusted neighbours
            for node_key in prev_local_trust.keys():
                if prev_local_trust[node_key] == 1:
                    trusted_neighbours.append(node_key)
            # Collect and average trusted neighbour opinion
            collected_trust = []
            for trusted_nei in trusted_neighbours:
                curr_trusted_nei = trusted_nei
                # if trust scores are not available, assume trusted
                nei_trust = prev_global_trust[trusted_nei].get(target_node, DEFAULT_NEI_TRUST)
                collected_trust.append(nei_trust)
            avg_trust = sum(collected_trust) / len(collected_trust) if len(collected_trust) > 0 else 0
            logging.info("[SentinelGlobal.get_trusted_neighbour_opinion] Avg. trusted neighbour opinion for node {}: {}"
                         .format(target_node, avg_trust))
            mapping = {f'Global Trust {target_node}': avg_trust}
            self.learner.logger.log_metrics(metrics=mapping, step=self.learner.logger.global_step)
        except KeyError as e:
            # There is a general memory issue with Fedstellar, where messages seem to be dropped
            logging.warning(
                "[SentinelGlobal.get_trusted_neighbour_opinion] KeyError for round {}, self.node_name: {}, curr_trusted_nei: {} target_node: {}"
                .format(prev_round, self.node_name, curr_trusted_nei, target_node))
            logging.warning("[SentinelGlobal.get_trusted_neighbour_opinion] With global trust: {}".format(self.global_trust[prev_round]))
        return avg_trust

    def evaluate_neighbour_model(self, model: OrderedDict, nodes: List[str], metrics: ModelMetrics):

        # TODO probably push down evaluation down to aggregation, since not all trust scores are available here
        model_params = model
        tmp_model = copy.deepcopy(self.learner.latest_model)
        tmp_model.load_state_dict(model_params)
        val_loss, val_acc = self.learner.validate_neighbour_no_pl2(tmp_model)
        tmp_model = None

        local_params = self.learner.get_parameters()
        cos_similarity: float = cosine_similarity(local_params, model_params)


        metrics.cosine_similarity = cos_similarity
        metrics.validation_loss = val_loss
        metrics.validation_accuracy = val_acc

        # Count number of evaluated models
        self.num_evals += 1
        mapping = {f'Models Evaluated': self.num_evals}
        self.learner.logger.log_metrics(metrics=mapping, step=self.learner.logger.global_step)

        return model, nodes, metrics

    def add_neighbour_trust(self, model: OrderedDict, nodes: List[str], metrics: ModelMetrics):
        neighbor_opinions = metrics.global_trust
        for round_key in neighbor_opinions.keys():
            for node_key in neighbor_opinions[round_key]:
                # local trust is added during aggregation
                if node_key != self.node_name:
                    try:
                        self.global_trust[round_key][node_key] = neighbor_opinions[round_key][node_key]
                    except:
                        for nei_key in self.neighbor_keys:
                            # There seems to be an issue with Fedstellar such that incomplete messages arrive
                            logging.warning("[SentinelGlobal.add_model]: Received incomplete neighbour trust. Fixing with DEFAULT_NEI_TRUST")
                            self.global_trust[round_key][node_key][nei_key] = DEFAULT_NEI_TRUST

    def add_model(self, model: OrderedDict, nodes: List[str], metrics: ModelMetrics):
        # No contributors (diffusion at Round 0)
        for node_key in nodes:
            self.neighbor_keys.add(node_key)

        if nodes is None:
            super().add_model(model=model, nodes=[], metrics=metrics)

        logging.info("[SentinelGlobal.add_model] Computing metrics for node(s): {}".format(nodes))
        self.add_neighbour_trust(model, nodes, metrics)

        # Step 0: Check whether the model should be evaluated based on global trust
        if self.agg_round > self.active_round:
            for node_key in nodes:
                avg_global_trust = self.get_trusted_neighbour_opinion(node_key)
                # Caveat: the node always trusts itself
                if avg_global_trust < TRUST_THRESHOLD and node_key != self.node_name:
                    # metrics.cosine_similarity = 0  # thereby the model will be removed by cosine filtering
                    logging.info("[SentinelGlobal.add_model] Removing node(s) {} since not trusted".format(node_key))
                else:
                    model, nodes, metrics = self.evaluate_neighbour_model(model, nodes, metrics)
        else:
            model, nodes, metrics = self.evaluate_neighbour_model(model, nodes, metrics)
        # model, nodes, metrics = self.evaluate_neighbour_model(model, nodes, metrics)
        super().add_model(model=model, nodes=nodes, metrics=metrics)
        # logging.info("[SentinelGlobal.add_model] New trust scores: {}".format(self.global_trust))

    def clear(self):
        """
        Clear all for a new aggregation.
        """
        if self.agg_round > self.active_round:
            pprint.pprint(self.global_trust[self.agg_round - 2])

            df_trust = pd.DataFrame.from_dict({(i, j): self.global_trust[i][j]
                                               for i in self.global_trust.keys()
                                               for j in self.global_trust[i].keys()},
                                              orient='index')
            # df_trust.tail()
            # self.learner.logger.log_text(key="Global Trust", dataframe=df_trust, step=self.logger.local_step)

        observers = self.get_observers()
        next_round = self.agg_round + 1
        prev_global_trust = copy.deepcopy(self.global_trust)
        self.__init__(node_name=self.node_name,
                      config=self.config,
                      logger=self.logger,
                      learner=self.learner,
                      agg_round=next_round,
                      global_trust=prev_global_trust,
                      num_evals=self.num_evals,
                      neighbor_keys=self.neighbor_keys)
        for o in observers:
            self.add_observer(o)

    def aggregate(self, models):

        logging.info("[SentinelGlobal]: Aggregation round {}".format(self.agg_round))
        # Check if there are models to aggregate
        if len(models) == 0:
            logging.warning("[SentinelGlobal] Trying to aggregate models when there is no models")
            return None

        # Log model metrics
        for node_key in models.keys():
            if node_key != self.node_name:
                metrics: ModelMetrics = models[node_key][1]
                mapping = {f'val_loss_{node_key}': metrics.validation_loss,
                           f'cos_{node_key}': metrics.cosine_similarity}
                self.learner.logger.log_metrics(metrics=mapping, step=self.learner.logger.global_step)

        # The model of the aggregator serves as a trusted reference
        my_model = models.get(self.node_name)  # change
        if my_model is None:
            logging.error("[SentinelGlobal] Trying to aggregate models when bootstrap is not available")
            return None

        # Step 1: Evaluate cosine similarity
        filtered_models = filter_models_by_cosine(models, COSINE_FILTER_THRESHOLD)
        malicious_by_cosine = models.keys() - filtered_models.keys()
        if len(filtered_models) == 0:
            logging.warning("[SentinelGlobal]: No more models to aggregate after filtering!")
            for model_key in models.keys():
                if model_key != self.node_name:
                    self.global_trust[self.agg_round][self.node_name][model_key] = 0
                metrics: ModelMetrics = models[model_key][1]
                print(f'Filtered model {model_key} with metrics {metrics}')

            return models.get(self.node_name)[0]

        for node_key in malicious_by_cosine:
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
        malicious_by_loss = {key for key, loss in mapped_loss.items() if loss == 0}

        logging.info("[SentinelGlobal]: Loss metrics: {}".format(loss))
        logging.info("[SentinelGlobal]: Loss mapped metrics: {}".format(mapped_loss))
        logging.info("[SentinelGlobal]: Cos metrics: {}".format(cos))

        # Step 3: Normalise the (remaining) untrusted models
        models_to_aggregate = {k: filtered_models[k] for k in filtered_models.keys() - {self.node_name}}
        normalised_models = {}
        for key in models_to_aggregate.keys():
            normalised_models[key] = normalise_layers(models_to_aggregate[key], my_model)
        normalised_models[self.node_name] = my_model

        # Create a Zero Model
        accum = (list(normalised_models.values())[-1][0]).copy()
        for layer in accum:
            accum[layer] = torch.zeros_like(accum[layer])

        # Aggregate
        total_mapped_loss: float = sum(mapped_loss.values())
        logging.info("[SentinelGlobal]: Total mapped loss: {}".format(total_mapped_loss))
        for node, message in normalised_models.items():
            client_model = message[0]
            for layer in client_model:
                accum[layer] = accum[layer] + client_model[layer] * mapped_loss[node]
                mapping = {f'agg_weight_{node}': mapped_loss[node] / total_mapped_loss}
                self.learner.logger.log_metrics(metrics=mapping, step=0)

        # Normalize accumulated model wrt loss
        for layer in accum:
            accum[layer] = accum[layer] / total_mapped_loss

        malicious = malicious_by_cosine.union(malicious_by_loss)
        logging.info("[SentinelGlobal]: Tagged as malicious: {}".format(list(malicious)))

        # Store global trust scores
        for node_key in models.keys():
            # The aggregator has already proceeded to the next round, thus
            if node_key in malicious and node_key != self.node_name:
                self.global_trust[self.agg_round][self.node_name][node_key] = 0
            else:
                self.global_trust[self.agg_round][self.node_name][node_key] = 1
        # logging.info("[SentinelGlobal]: Global trust scores after adding: {}".format(self.global_trust))

        return accum
