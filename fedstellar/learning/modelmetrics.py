from dataclasses import dataclass
from typing import List, OrderedDict


@dataclass
class ModelMetrics:
    num_samples: int = 0
    validation_loss: float = 0
    cosine_similarity: float = 0
    validation_accuracy: float = 0

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