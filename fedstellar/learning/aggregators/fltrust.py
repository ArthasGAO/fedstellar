#
# This file is part of the fedstellar framework (see https://github.com/enriquetomasmb/fedstellar).
# Copyright (c) 2022 Enrique Tomás Martínez Beltrán.
#


import logging
import pickle
import copy
from typing import OrderedDict, Optional, List

import torch
from statistics import mean
from fedstellar.learning.aggregators.aggregator import Aggregator
from fedstellar.learning.aggregators.helper import cosine_similarity
from fedstellar.learning.modelmetrics import ModelMetrics


def normalise_layers(untrusted_models, trusted_model):
    bootstrap = trusted_model[0]
    trusted_norms = dict([k, torch.norm(bootstrap[k].data.view(-1))] for k in bootstrap.keys())

    normalised_models = copy.deepcopy(untrusted_models)
    for client, message in untrusted_models.items():
        state_dict = message[0]
        norm_state_dict = state_dict.copy()
        for layer in state_dict:
            layer_norm = torch.norm(state_dict[layer].data.view(-1))
            scaling_factor = trusted_norms[layer] / layer_norm
            # logging.info("Scaling client {} layer {} with factor {}".format(client, layer, scaling_factor))
            normalised_layer = torch.mul(state_dict[layer], scaling_factor)
            normalised_models[client][0][layer] = normalised_layer

    return normalised_models


class FlTrust(Aggregator):
    """
    Federated Averaging (FedAvg) [McMahan et al., 2016]
    Paper: https://arxiv.org/abs/1602.05629
    """

    def __init__(self, node_name="unknown", config=None, logger=None, learner=None, agg_round=0):
        super().__init__(node_name, config, logger, learner, agg_round)
        self.config = config
        self.role = self.config.participant["device_args"]["role"]
        logging.info("[FLTrust] My config is {}".format(self.config))

    def add_model(self, model: OrderedDict, nodes: List[str], metrics: ModelMetrics):

        logging.info("[Sentinel.add_model] Computing metrics for node(s): {}".format(nodes))

        model_params = model

        local_params = self.learner.get_parameters()
        cos_similarity: float = cosine_similarity(local_params, model_params)

        if nodes is not None:
            for node in nodes:
                if node != self.node_name:
                    mapping = {f'cos_{node}': cos_similarity}
                    self.learner.logger.log_metrics(metrics=mapping, step=self.learner.logger.global_step)

        metrics.cosine_similarity = cos_similarity

        logging.info("[Sentinel.add_model] Computed metrics for node(s): {}: --> {}".format(nodes, metrics))

        super().add_model(model=model, nodes=nodes, metrics=metrics)

    def aggregate(self, models):
        """
        TrimmedMean [Cao et al., 2022]
        Paper: https://arxiv.org/abs/2012.13995

         Args:
             models: Dictionary with the models (node: model,num_samples).
         """
        # Check if there are models to aggregate
        if len(models) == 0:
            logging.error("[FlTrust] Trying to aggregate models when there is no models")
            return None

        # The model of the aggregator serves as a trusted reference
        my_model = models.get(self.node_name)  # change
        if my_model is None:
            logging.error("[FlTrust] Own model as bootstrap is not available")
            return None

        untrusted_models = {k: models[k] for k in models.keys() - {self.node_name}}  # change

        # Normalise the untrusted models
        normalised_models = normalise_layers(untrusted_models, my_model)
        normalised_models[self.node_name] = my_model

        # Create a Zero Model
        accum = (list(models.values())[-1][0]).copy()
        for layer in accum:
            accum[layer] = torch.zeros_like(accum[layer])

        # Aggregate
        similarities = []
        for client, message in normalised_models.items():
            client_model = message[0]
            metrics: ModelMetrics = message[1]
            similarities.append(metrics.cosine_similarity)
            for layer in client_model:
                accum[layer] = accum[layer] + client_model[layer] * metrics.cosine_similarity

        # Normalize Accum
        avg_similarity = mean(similarities)
        for layer in accum:
            accum[layer] = accum[layer] / avg_similarity

        logging.info("[FlTrust.aggregate] Aggregated model with weights: similarities={}".format(similarities))

        return accum
