# 
# This file is part of the fedstellar framework (see https://github.com/enriquetomasmb/fedstellar).
# Copyright (c) 2023 Enrique Tomás Martínez Beltrán.
# 


import logging
from typing import Dict

import torch

from fedstellar.learning.aggregators.aggregator import Aggregator
from fedstellar.learning.modelmetrics import ModelMetrics


class FedAvg(Aggregator):
    """
    Federated Averaging (FedAvg) [McMahan et al., 2016]
    Paper: https://arxiv.org/abs/1602.05629
    """

    def __init__(self, node_name="unknown", config=None, logger=None, learner=None, agg_round=0):
        super().__init__(node_name, config, logger, learner, agg_round)

        self.config = config
        self.role = self.config.participant["device_args"]["role"]
        logging.info("[FedAvg] My config is {}".format(self.config))

    def aggregate(self, models):
        """
        Ponderated average of the models.

        Args:
            models: Dictionary with the models (node: model, metrics).
        """
        # Check if there are models to aggregate
        if len(models) == 0:
            logging.error(
                "[FedAvg] Trying to aggregate models when there is no models"
            )
            return None

        """
        if len(models) >= 3:
            with open('/Users/janosch/Desktop/models.pk', 'wb') as handle:
                logging.info("Saving received models")
                pickle.dump(models, handle, protocol=pickle.HIGHEST_PROTOCOL)
        """

        for node, msg in models.items():
            params = msg[0]
            metrics: ModelMetrics = msg[1]
            logging.info("FedAvg model metrics: {}".format(metrics))

        models = list(models.values())

        # Total Samples
        y: ModelMetrics
        total_samples = sum([y.num_samples for _, y in models])

        if total_samples == 0:
            logging.error("FedAvg: Did not receive sample metrics")
        else:
            logging.info("[FedAvg.aggregate]: aggregating with {} samples".format(total_samples))

        # Create a Zero Model
        accum = (models[-1][0]).copy()
        for layer in accum:
            accum[layer] = torch.zeros_like(accum[layer])

        # Add weighteds models
        logging.info("[FedAvg.aggregate] Aggregating models: num={}".format(len(models)))
        metrics: ModelMetrics
        for model, metrics in models:
            for layer in model:
                accum[layer] = accum[layer] + model[layer] * metrics.num_samples

        # Normalize Accum
        for layer in accum:
            accum[layer] = accum[layer] / total_samples

        return accum
