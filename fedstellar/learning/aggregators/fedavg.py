# 
# This file is part of the fedstellar framework (see https://github.com/enriquetomasmb/fedstellar).
# Copyright (c) 2022 Enrique Tomás Martínez Beltrán.
# 


import logging

import torch

from fedstellar.learning.aggregators.aggregator import Aggregator


class FedAvg(Aggregator):
    """
    Federated Averaging (FedAvg) [McMahan et al., 2016]
    Paper: https://arxiv.org/abs/1602.05629
    """

    def __init__(self, node_name="unknown", config=None):
        super().__init__(node_name, config)
        self.config = config
        self.role = self.config.participant["device_args"]["role"]
        logging.info("[FedAvg] My config is {}".format(self.config))

    def aggregate(self, models):
        """
        Ponderated average of the models.

        Args:
            models: Dictionary with the models (node: model,num_samples).
        """
        # Check if there are models to aggregate
        if len(models) == 0:
            logging.error(
                "[FedAvg] Trying to aggregate models when there is no models"
            )
            return None

        models = list(models.values())

        # Total Samples
        total_samples = sum([y for _, y in models])

        # Create a Zero Model
        accum = (models[-1][0]).copy()
        for layer in accum:
            accum[layer] = torch.zeros_like(accum[layer])

        # Add weighteds models
        logging.info("[FedAvg.aggregate] Aggregating models: num={}".format(len(models)))
        for m, w in models:
            for layer in m:
                accum[layer] = accum[layer] + m[layer] * w

        # Normalize Accum
        for layer in accum:
            accum[layer] = accum[layer] / total_samples

        return accum
