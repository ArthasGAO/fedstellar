from dataclasses import dataclass
from typing import List, OrderedDict


@dataclass
class ModelMetrics:
    samples: int = 0
    loss: float = 0
    similarity: float = 0

"""
@dataclass
class ModelMessage:
    model: OrderedDict
    nodes: List[str]
    metrics: ModelMetrics

    def reset_personal_metrics(self):
        self.metrics.similarity = 0
        self.metrics.loss = 0
"""