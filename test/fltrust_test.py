from collections import OrderedDict

import torch
import copy
import numpy as np
from fedstellar.learning.aggregators.fedavg import FedAvg
from fedstellar.learning.aggregators.sentinel import Sentinel
from fedstellar.learning.pytorch.lightninglearner import LightningLearner
from fedstellar.learning.pytorch.mnist.models.mlp import MLP
from fedstellar.node import Node
from test.utils import set_test_settings

set_test_settings()


# ------------------- FlTrust ------------------------ #

def get_test_models(bs_scale=1, c1_scale=1, c2_scale=1):
    # Set up test data
    node_name = 'node1'

    tmp = np.ones((2, 2), dtype=np.uint8)
    layer1_base = torch.from_numpy(tmp)

    tmp = np.ones((3, 3), dtype=np.uint8)
    layer2_base = torch.from_numpy(tmp)

    bootstrap_layer1 = torch.mul(layer1_base, bs_scale)
    bootstrap_layer2 = torch.mul(layer2_base, bs_scale)
    bootstrap = {'layer1': bootstrap_layer1, 'layer2': bootstrap_layer2}
    client1_layer1 = torch.mul(layer1_base, c1_scale)
    client1_layer2 = torch.mul(layer2_base, c1_scale)
    client1_model = {'layer1': client1_layer1, 'layer2': client1_layer2}
    client2_layer1 = torch.mul(layer1_base, c2_scale)
    client2_layer2 = torch.mul(layer2_base, c2_scale)
    client2_model = {'layer1': client2_layer1, 'layer2': client2_layer2}
    untrusted_models = {'client1': [client1_model], 'client2': [client2_model]}

    return bootstrap, untrusted_models


def test_normalise_layers():
    bootstrap, untrusted_models = get_test_models(2, 1, 3)
    aggregator = FlTrust()
    actual_normalised_models = aggregator.normalise_layers(untrusted_models, bootstrap)

    actual_normalised_models = aggregator.normalise_layers(untrusted_models, bootstrap)
    for client, message in actual_normalised_models.items():
        layers = message[0]
        for layer in layers.values():
            assert torch.eq(layer, 2).all()


def test_cosine_similarity():
    bootstrap, untrusted_models = get_test_models(1, 2, -1)

    aggregator = FlTrust()
    similarities = aggregator.cosine_similarities(untrusted_models, bootstrap)

    assert similarities['client1'] > 0.9999  # Same but scaled
    assert similarities['client2'] < 0.0001  # Relu clipped
