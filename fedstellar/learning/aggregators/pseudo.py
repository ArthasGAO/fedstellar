#
# This file is part of the fedstellar framework (see https://github.com/enriquetomasmb/fedstellar).
# Copyright (c) 2022 Enrique Tomás Martínez Beltrán.
#


import logging
import pickle
import copy
import time

import torch
from statistics import mean

from typing import List, OrderedDict

from fedstellar.learning.aggregators.aggregator import Aggregator
from fedstellar.learning.aggregators.helper import cosine_similarity
from fedstellar.learning.modelmetrics import ModelMetrics


class PseudoAggregator(Aggregator):
    """
    Federated Averaging (FedAvg) [McMahan et al., 2016]
    Paper: https://arxiv.org/abs/1602.05629
    """

    def __init__(self, node_name="unknown", config=None, logger=None, learner=None, agg_round=0):
        super().__init__(node_name, config, logger, learner, agg_round)

        self.config = config
        self.role = self.config.participant["device_args"]["role"]
        logging.info("[PseudoAggregator] My config is {}".format(self.config))

    def add_model(self, model: OrderedDict, nodes: List[str], metrics: ModelMetrics):

        logging.info("[Sentinel.add_model] Computing metrics for node(s): {}".format(nodes))

        model_params = model

        tmp_model = copy.deepcopy(self.learner.latest_model)
        tmp_model.load_state_dict(model_params)
        val_loss, val_acc = self.learner.validate_neighbour_no_pl(tmp_model)
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

    def aggregate(self, models):

        if len(models) == 0:
            logging.error("[Sentinel] Trying to aggregate models when there is no models")
            return None

        # The model of the aggregator serves as a trusted reference
        my_model = models.get(self.node_name)  # change
        if my_model is None:
            logging.error("[Pseudo] Own model as bootstrap is not available")
            return None

        logging.info("[PseudoAggregator] Replace aggregate with own model")
        pseudo_accum = my_model[0]

        return pseudo_accum
