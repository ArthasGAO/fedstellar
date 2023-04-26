# 
# This file is part of the fedstellar framework (see https://github.com/enriquetomasmb/fedstellar).
# Copyright (c) 2022 Enrique Tomás Martínez Beltrán.
# 


import logging
import pickle
import copy
import torch
from statistics import mean
from fedstellar.learning.aggregators.aggregator import Aggregator


class FlTrust(Aggregator):
    """
    Federated Averaging (FedAvg) [McMahan et al., 2016]
    Paper: https://arxiv.org/abs/1602.05629
    """

    def __init__(self, node_name="unknown", config=None, logger=None):
        super().__init__(node_name, config, logger)
        self.config = config
        self.role = self.config.participant["device_args"]["role"]
        logging.info("[FLTrust] My config is {}".format(self.config))

    def cosine_similarities(self, untrusted_models, trusted_model):
        similarities = dict.fromkeys(untrusted_models.keys(), [])
        similarities[self.node_name] = [1]

        bootstrap = trusted_model[0]

        for client, message in untrusted_models.items():
            state_dict = message[0]
            client_similarities = []

            for layer in bootstrap:
                l1 = bootstrap[layer]
                l2 = state_dict[layer]
                cosine_similarity = torch.nn.CosineSimilarity(dim=l1.dim() - 1)
                # does it make sense to take mean from row-wise similarity?
                cos_mean = torch.mean(cosine_similarity(l1, l2)).mean()
                client_similarities.append(cos_mean)

            similarities[client] = client_similarities

        for client in similarities:
            cos = torch.Tensor(similarities[client])
            avg_cos = torch.mean(cos)
            relu_cos = torch.nn.functional.relu(avg_cos)
            similarities[client] = relu_cos.item()

        return similarities

    def normalise_layers(self, untrusted_models, trusted_model):
        bootstrap = trusted_model[0]
        trusted_norms = dict([k, torch.norm(bootstrap[k].data.view(-1))] for k in bootstrap.keys())

        normalised_models = copy.deepcopy(untrusted_models)
        for client, message in untrusted_models.items():
            state_dict = message[0]
            norm_state_dict = state_dict.copy()
            for layer in state_dict:
                layer_norm = torch.norm(state_dict[layer].data.view(-1))
                scaling_factor = trusted_norms[layer] / layer_norm
                normalised_layer = torch.mul(state_dict[layer], scaling_factor)
                normalised_models[client][0][layer] = normalised_layer

        return normalised_models

    def aggregate(self, models):
        """
         Krum selects one of the m local models that is similar to other models
         as the global model, the euclidean distance between two local models is used.

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

        # Compute cosine similarities for all neighboring models
        similarities = self.cosine_similarities(untrusted_models, my_model)

        # Normalise the untrusted models
        normalised_models = self.normalise_layers(untrusted_models, my_model)
        normalised_models[self.node_name] = my_model

        # Create a Zero Model
        accum = (list(models.values())[-1][0]).copy()
        for layer in accum:
            accum[layer] = torch.zeros_like(accum[layer])

        # Aggregate
        for client, message in normalised_models.items():
            client_model = message[0]
            for layer in client_model:
                accum[layer] = accum[layer] + client_model[layer] * similarities[client]

        # Normalize Accum
        avg_similarity = mean(similarities.values())
        total_similarity = sum(similarities.values())
        for layer in accum:
            accum[layer] = accum[layer] / total_similarity

        logging.info("[FlTrust.aggregate] Aggregated model at host {} with weights: similarities={} "
                     "and avg. similarity: {}"
                     .format(self.node_name, similarities, avg_similarity))
        if self.logger is not None:
            key = "similarities"
            columns = list(similarities.keys())
            data = [list(similarities.values())]
            self.logger.log_text(key=key, columns=columns, data=data, step=0)

        return accum
