# 
# This file is part of the fedstellar framework (see https://github.com/enriquetomasmb/fedstellar).
# Copyright (c) 2022 Chao Feng.
# 


import logging
import torch
import numpy
from statistics import mean
import copy
from torch.nn.functional import cosine_similarity
from fedstellar.learning.aggregators.aggregator import Aggregator


class Krum(Aggregator):
    """
    Krum [Peva Blanchard et al., 2017]
    Paper: https://papers.nips.cc/paper/2017/hash/f4b9ec30ad9f68f89b29639786cb62ef-Abstract.html
    """

    def __init__(self, node_name="unknown", config=None):
        super().__init__(node_name, config)
        self.config = config
        self.role = self.config.participant["device_args"]["role"]
        logging.info("[Krum] My config is {}".format(self.config))

    def aggregate(self, models):
        """
        Krum selects one of the m local models that is similar to other models 
        as the global model, the euclidean distance between two local models is used.

        Args:
            models: Dictionary with the models (node: model,num_samples).
        """
        # Check if there are models to aggregate
        if len(models) == 0:
            logging.error(
                "[Krum] Trying to aggregate models when there is no models"
            )
            return None

        # The aggregator's update is considered trusted
        bootstrap = models.get(self.node_name)[0]  # change

        untrusted_models = {k: models[k] for k in models.keys() - {self.node_name}}  # change
        similarities = dict.fromkeys(untrusted_models.keys(), [])
        similarities[self.node_name] = [1]

        for client, message in untrusted_models.items():
            state_dict = message[0]
            for layer in bootstrap:
                l1 = bootstrap[layer]
                l2 = state_dict[layer]

                cosine_similarity = torch.nn.CosineSimilarity(dim=l1.dim() - 1)
                # does it make sense to take mean from row-wise similarity?
                cos_mean = torch.mean(cosine_similarity(l1, l2)).mean()
                similarities[client].append(cos_mean.item())

        for client in similarities:
            cos = torch.Tensor(similarities[client])
            avg_cos = torch.mean(cos)
            relu_cos = torch.nn.functional.relu(avg_cos)
            similarities[client] = relu_cos.item()

        # norm bounding
        normalised_models = copy.deepcopy(untrusted_models)
        # calculate layer-wise norm of flattened layers
        trusted_norms = dict([k, torch.norm(bootstrap[k].data.view(-1))] for k in bootstrap.keys())

        for client, message in untrusted_models.items():
            state_dict = message[0]
            norm_state_dict = state_dict.copy()
            for layer in state_dict:
                layer_norm = torch.norm(state_dict[layer].data.view(-1))
                scaling_factor = trusted_norms[layer] / layer_norm
                normalised_layer = torch.mul(state_dict[layer], scaling_factor)
                normalised_models[client][0][layer] = normalised_layer

        models = list(models.values())

        normalised_models[self.node_name] = models.get(self.node_name)
        normed_models = list(normalised_models.values())

        # Create a Zero Model
        accum = (normed_models[-1][0]).copy()
        for layer in accum:
            accum[layer] = torch.zeros_like(accum[layer])

        for client, message in untrusted_models.items():
            model = message[0]
            for layer in model:
                accum[layer] = accum[layer] + model[layer] * similarities[client]

        # Normalize Accum
        total_similarity = mean(similarities.values())
        print(total_similarity)
        for layer in accum:
            accum[layer] = accum[layer] / total_similarity

        print(accum)

        logging.info("[Krum.aggregate] Aggregated model: accum={}".format(accum))

        return accum
