import logging
from typing import OrderedDict, List, Optional
import copy
import torch


def cosine_similarity(trusted_model: OrderedDict, untrusted_model: OrderedDict) -> Optional[float]:
    if trusted_model is None or untrusted_model is None:
        logging.info("Cosine similarity cannot be computed due to missing model")
        return None

    layer_similarities: List = []

    for layer in trusted_model:
        l1 = trusted_model[layer].to('cpu')
        l2 = untrusted_model[layer].to('cpu')
        cos = torch.nn.CosineSimilarity(dim=l1.dim() - 1)
        cos_mean = torch.mean(cos(l1.float(), l2.float())).mean()
        layer_similarities.append(cos_mean)

    cos = torch.Tensor(layer_similarities)
    avg_cos = torch.mean(cos)
    relu_cos = torch.nn.functional.relu(avg_cos)
    result = relu_cos.item()

    return result


def normalise_layers(untrusted_model, trusted_model):
    bootstrap = trusted_model[0]
    trusted_norms = dict([k, torch.norm(bootstrap[k].data.view(-1).float())] for k in bootstrap.keys())

    normalised_model = copy.deepcopy(untrusted_model)

    state_dict = untrusted_model[0]
    for layer in state_dict:
        layer_norm = torch.norm(state_dict[layer].data.view(-1).float())
        scaling_factor = min(layer_norm / trusted_norms[layer], 1)
        logging.debug("[Aggregator.normalise_layers()] Layer: {} ScalingFactor {}".format(layer, scaling_factor))
        # logging.info("Scaling client {} layer {} with factor {}".format(client, layer, scaling_factor))
        normalised_layer = torch.mul(state_dict[layer], scaling_factor)
        normalised_model[0][layer] = normalised_layer

    return normalised_model
