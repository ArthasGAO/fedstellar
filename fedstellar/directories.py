import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))  # Parent directory where is the fedstellar module

from fedstellar.learning.aggregators.fedavg import FedAvg
import pytorch_lightning
import torch
import inspect

from fedstellar.learning.aggregators.aggregator import Aggregator
from fedstellar.learning.modelmetrics import ModelMetrics

aggregator = FedAvg()

print(inspect.isclass(FedAvg))

print(type(aggregator))