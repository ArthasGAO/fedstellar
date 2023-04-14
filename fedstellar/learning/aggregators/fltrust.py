# 
# This file is part of the fedstellar framework (see https://github.com/enriquetomasmb/fedstellar).
# Copyright (c) 2022 Enrique Tomás Martínez Beltrán.
# 


import logging
import pickle

import torch

from fedstellar.learning.aggregators.aggregator import Aggregator


class FlTrust(Aggregator):
    """
    Federated Averaging (FedAvg) [McMahan et al., 2016]
    Paper: https://arxiv.org/abs/1602.05629
    """

    def __init__(self, node_name="unknown", config=None):
        super().__init__(node_name, config)
        self.config = config
        self.role = self.config.participant["device_args"]["role"]
        logging.info("[FLTrust] My config is {}".format(self.config))


    def aggregate(self, models):
        """
        Ponderated average of the models.

        Args:
            models: Dictionary with the models (node: model,num_samples).
        """

        logging.debug("[FlTrust.aggregate] Self {}. Keys {} \n, ".format(self.node_name, models.keys()))

        # Check if there are models to aggregate
        if len(models) == 0:
            logging.error("[FlTrust.aggregate] Trying to aggregate models when there is no models")
            return None

        # Extract node's own local model as trusted reference
        trusted = models.get(self.node_name)
        models_untrusted = {k: models[k] for k in set(list(models.keys())) - set(self.node_name)}
        if trusted is None:
            logging.error("[FlTrust.aggregate] No local model available at {}".format(self.node_name))
            return None

        logging.debug("trust model: {}".format(trusted))



        cosine_similarity = torch.nn.CosineSimilarity(dim=0, eps=1e-6)

        # Store data (serialize)
        with open('/Users/janosch/Desktop/models.pk', 'wb') as handle:
            pickle.dump(models, handle, protocol=pickle.HIGHEST_PROTOCOL)

        models = list(models.values())

        # Total Samples
        total_samples = sum([y for _, y in models])

        # Create a Zero Model
        accum = (models[-1][0]).copy()
        for layer in accum:
            accum[layer] = torch.zeros_like(accum[layer])

        # Add weighteds models
        logging.info("[FlTrust.aggregate] Aggregating models: num={}".format(len(models)))
        for m, w in models:
            for layer in m:
                accum[layer] = accum[layer] + m[layer] * w

        # Normalize Accum
        for layer in accum:
            accum[layer] = accum[layer] / total_samples

        return accum
