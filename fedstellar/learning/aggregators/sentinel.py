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

from typing import List, Dict, OrderedDict, Optional

from pytorch_lightning.loggers import wandb

from fedstellar.learning.aggregators.aggregator import Aggregator
from fedstellar.learning.modelmetrics import ModelMetrics
from fedstellar.learning.pytorch.lightninglearner import LightningLearner


def cosine_similarity(trusted_model: OrderedDict, untrusted_model: OrderedDict) -> Optional[float]:
    if trusted_model is None or untrusted_model is None:
        logging.info("Cosine similarity cannot be computed due to missing model")
        return None

    layer_similarities: List = []

    for layer in trusted_model:
        l1 = trusted_model[layer]
        l2 = untrusted_model[layer]
        cos = torch.nn.CosineSimilarity(dim=l1.dim() - 1)
        cos_mean = torch.mean(cos(l1, l2)).mean()
        layer_similarities.append(cos_mean)

    cos = torch.Tensor(layer_similarities)
    avg_cos = torch.mean(cos)
    relu_cos = torch.nn.functional.relu(avg_cos)
    result = relu_cos.item()

    return result


def filter_models(models: Dict, threshold: float) -> Dict:
    filtered: Dict = {}
    for node, msg in models.items():
        params = msg[0]
        metrics: ModelMetrics = msg[1]
        if metrics.cosine_similarity > threshold:
            filtered[node] = msg
    return filtered


# hoooold the dooor
def map_loss(loss) -> float:
    return math.exp(-loss)


# boy gets flipped by label-flip
def map_loss2(loss) -> float:
    return 1/(abs(loss) + 1)


class Sentinel(Aggregator):
    """
    Federated Averaging (FedAvg) [McMahan et al., 2016]
    Paper: https://arxiv.org/abs/1602.05629
    """

    def __init__(self, node_name="unknown", config=None, logger=None, learner=None):
        super().__init__(node_name, config, logger)
        self.config = config
        self.role = self.config.participant["device_args"]["role"]
        logging.info("[FLTrust] My config is {}".format(self.config))
        self.logger = logger
        self.learner = learner

        logging.info("Received logger: {}".format(logger))
        logging.info("Received learner: {}".format(learner))

    def cosine_similarities_last_layer(self, untrusted_models, trusted_model):
        similarities = dict.fromkeys(untrusted_models.keys())
        similarities[self.node_name] = 1

        bootstrap = trusted_model[0]

        for client, message in untrusted_models.items():
            state_dict = message[0]

            l1 = bootstrap[next(reversed(bootstrap))]
            l2 = state_dict[next(reversed(state_dict))]

            l1 = l1.data.view(-1)
            l2 = l2.data.view(-1)

            cosine_similarity = torch.nn.CosineSimilarity(dim=l1.dim() - 1)
            # does it make sense to take mean from row-wise similarity?
            cos = cosine_similarity(l1, l2)
            relu_cos = torch.nn.functional.relu(cos)

            similarities[client] = relu_cos.item()

        return similarities

    def cosine_similarities(self, untrusted_models, trusted_model):
        similarities = dict.fromkeys(untrusted_models.keys())
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

        logging.info("[FlTrust.aggregate] Analysed layers at {} for cosine: {}"
                     .format(self.node_name, similarities))

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
                # logging.info("Scaling client {} layer {} with factor {}".format(client, layer, scaling_factor))
                normalised_layer = torch.mul(state_dict[layer], scaling_factor)
                normalised_models[client][0][layer] = normalised_layer

        return normalised_models

    def aggregate(self, models):

        # logging.info("Current logger step: {}".format(self.logger.global_step))
        # logging.info("Current learner step: {}".format(self.learner.logger.global_step))

        """
         Krum selects one of the m local models that is similar to other models
         as the global model, the euclidean distance between two local models is used.

         Args:
             models: Dictionary with the models (node: model,num_samples).
             step: current training step (used for logging utility)
         """
        # Check if there are models to aggregate
        if len(models) == 0:
            logging.error("[FlTrust] Trying to aggregate models when there is no models")
            return None

        """
        if len(models) >= 2:
            filename = "models" + self.node_name + ".pickle"
            with open('/Users/janosch/Desktop/models.pickle', 'wb') as handle:
                pickle.dump(filename, handle, protocol=pickle.HIGHEST_PROTOCOL)
        """

        """        
        for client, msg in models.items():
            model = msg[0]
            loss, metric = self.learner.evaluate_neighbour(model)
            logging.info("Eval at {}: Loss {}, Metric: {}".format(client, loss, metric))
        """

        # The model of the aggregator serves as a trusted reference
        my_model = models.get(self.node_name)  # change
        if my_model is None:
            logging.error("[FlTrust] Own model as bootstrap is not available")
            return None

        # untrusted_models = {k: models[k] for k in models.keys() - {self.node_name}}  # change

        loss: Dict = {}
        cos: Dict = {}
        for node, msg in models.items():
            params = msg[0]
            metrics: ModelMetrics = msg[1]
            logging.info("FlTrust metrics: {}".format(metrics))
            loss[node] = metrics.validation_loss
            cos[node] = metrics.cosine_similarity
        logging.info("FlTrust: Loss metrics: {}, cosine metrics: {}".format(loss, cos))

        filtered_models = filter_models(models, 0)
        if len(filtered_models) == 0:
            logging.warning("FlTrust: No more models to aggregate after filtering!")
            return None

        # Create a Zero Model
        accum = (list(models.values())[-1][0]).copy()
        for layer in accum:
            accum[layer] = torch.zeros_like(accum[layer])

        # Aggregate
        total_mapped_loss: float = float(0)
        for client, message in models.items():
            client_model = message[0]
            metrics = message[1]
            mapped_loss = map_loss(metrics.validation_loss)
            logging.warning("FlTrust: Agg model {} with loss {}, mapped loss {}"
                            .format(client, metrics.validation_loss, mapped_loss))
            total_mapped_loss += mapped_loss
            for layer in client_model:
                accum[layer] = accum[layer] + client_model[layer] * mapped_loss

        logging.warning("FlTrust: Total mapped loss: {}".format(total_mapped_loss))
        # Normalize accumulated model wrt loss
        for layer in accum:
            accum[layer] = accum[layer] / total_mapped_loss

        return accum
