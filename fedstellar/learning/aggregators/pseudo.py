#
# This file is part of the fedstellar framework (see https://github.com/enriquetomasmb/fedstellar).
# Copyright (c) 2022 Enrique Tomás Martínez Beltrán.
#


import logging
import pickle
import copy
import time

import torch
from statistics import mean

from typing import List

from fedstellar.learning.aggregators.aggregator import Aggregator


class PseudoAggregator(Aggregator):
    """
    Federated Averaging (FedAvg) [McMahan et al., 2016]
    Paper: https://arxiv.org/abs/1602.05629
    """

    def __init__(self, node_name="unknown", config=None, logger=None):
        super().__init__(node_name, config, logger)
        self.config = config
        self.role = self.config.participant["device_args"]["role"]
        logging.info("[PseudoAggregator] My config is {}".format(self.config))

    def aggregate(self, models):
        logging.info("[PseudoAggregator] Replace aggregate with own model")
        pseudo_accum = models.get(self.node_name)[0]

        return pseudo_accum
