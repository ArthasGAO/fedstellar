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

MIN_MAPPED_LOSS = float(0.01)
COSINE_FILTER_THRESHOLD = float(0.5)


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
        else:
            logging.info("[Sentinel] Filtering model due to low cos similarity {}".format(node))
    return filtered


# hoooold the dooor
def map_loss(loss) -> float:
    mapped_loss = math.exp(-loss)
    if (math.isnan(mapped_loss)) | (mapped_loss < MIN_MAPPED_LOSS):
        # logging.error("[Sentinel] Mapped loss is invalid! Return small value instead")
        return float(MIN_MAPPED_LOSS)
    return mapped_loss


# maps loss of neighbours relative distance to reference model
# inverse own loss serves as damping factors to allow more aggregation in early FL rounds
def map_loss_distance(loss: float, my_loss: float) -> float:
    loss_dist = loss - my_loss
    # prevent division by zero
    if my_loss <= MIN_MAPPED_LOSS:
        my_loss = MIN_MAPPED_LOSS
    k = 1 / my_loss
    mapped_distance_loss = math.exp(-k * loss_dist)
    if (math.isnan(mapped_distance_loss)) | (mapped_distance_loss < MIN_MAPPED_LOSS):
        # logging.error("[Sentinel] Mapped loss is invalid! Return small value instead")
        return float(0)
    return mapped_distance_loss


# boy gets flipped by label-flip
def map_loss2(loss) -> float:
    return 1 / (abs(loss) + 1)


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


class Sentinel(Aggregator):
    """
    Sentinel
        Custom aggregation method based on cosine similarity, loss distance and normalisation.
    """

    def __init__(self, node_name="unknown", config=None, logger=None, learner=None, model_struct=None):
        super().__init__(node_name, config, logger, learner)
        self.config = config
        self.role = self.config.participant["device_args"]["role"]
        logging.info("[Sentinel] My config is {}".format(self.config))
        self.logger = logger
        self.learner = learner
        self.model_struct = model_struct

    def aggregate(self, models):

        # logging.info("Current logger step: {}".format(self.logger.global_step))
        # logging.info("Current learner step: {}".format(self.learner.logger.global_step))

        """
         Krum selects one of the m local models that is similar to other models
         as the global model, the euclidean distance between two local models is used.

         Args:
             models: Dictionary with the models (node: model,num_samples).
         """
        # Check if there are models to aggregate

        if len(models) == 0:
            logging.warning("[Sentinel] Trying to aggregate models when there is no models")
            return None

        # The model of the aggregator serves as a trusted reference
        my_model = models.get(self.node_name)  # change
        if my_model is None:
            logging.warning("[Sentinel] Trying to aggregate models when bootstrap is not available")
            return None
        my_loss = my_model[1].validation_loss

        """
        if len(models) >= 2:
            filename = "models" + self.node_name + ".pickle"
            with open('/Users/janosch/Desktop/models.pickle', 'wb') as handle:
                pickle.dump(filename, handle, protocol=pickle.HIGHEST_PROTOCOL)
        """

        """
        for client, msg in models.items():
            params = msg[0]
            # learner sometimes none, why?
            loss, metric = self.learner.validate_neighbour(params)
            logging.info("Eval at {}: Loss {}, Metric: {}".format(client, loss, metric))
        """

        loss: Dict = {}
        mapped_loss: Dict = {}
        cos: Dict = {}
        for node, msg in models.items():
            params = msg[0]
            metrics: ModelMetrics = msg[1]
            loss[node] = metrics.validation_loss
            mapped_loss[node] = map_loss_distance(metrics.validation_loss, my_loss)
            cos[node] = metrics.cosine_similarity
        logging.info("[Sentinel]: Loss metrics: {}".format(loss))
        logging.info("[Sentinel]: Loss mapped metrics: {}".format(mapped_loss))
        logging.info("[Sentinel]: Cos metrics: {}".format(cos))

        malicous_by_loss = {key for key, loss in mapped_loss.items() if loss == 0}

        filtered_models = filter_models(models, COSINE_FILTER_THRESHOLD)
        malicous_by_cosine = models.keys() - filtered_models.keys()
        if len(filtered_models) == 0:
            logging.warning("Sentinel: No more models to aggregate after filtering!")
            return None

        # Create a Zero Model
        accum = (list(models.values())[-1][0]).copy()
        for layer in accum:
            accum[layer] = torch.zeros_like(accum[layer])

        # Aggregate
        total_mapped_loss: float = sum(mapped_loss.values())
        # logging.info("Sentinel: Total mapped loss: {}".format(total_mapped_loss))
        for node, message in models.items():
            client_model = message[0]
            for layer in client_model:
                accum[layer] = accum[layer] + client_model[layer] * mapped_loss[node]

        # Normalize accumulated model wrt loss
        for layer in accum:
            accum[layer] = accum[layer] / total_mapped_loss

        malicious = malicous_by_cosine.union(malicous_by_loss)
        logging.info("Sentinel: Tagged as malicious: {}".format(list(malicious)))

        return accum

    """
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
    """

    """
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

        logging.info("[Sentinel.aggregate] Analysed layers at {} for cosine: {}"
                     .format(self.node_name, similarities))

        for client in similarities:
            cos = torch.Tensor(similarities[client])
            avg_cos = torch.mean(cos)
            relu_cos = torch.nn.functional.relu(avg_cos)
            similarities[client] = relu_cos.item()

        return similarities
    """
