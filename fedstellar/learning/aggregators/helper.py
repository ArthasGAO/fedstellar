import logging
from typing import OrderedDict, List, Optional

import torch


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
