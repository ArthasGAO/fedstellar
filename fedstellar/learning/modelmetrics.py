from dataclasses import dataclass, field
from typing import List, OrderedDict, Dict


@dataclass
class ModelMetrics:
    num_samples: int = 0
    validation_loss: float = 0
    cosine_similarity: float = 0
    validation_accuracy: float = 0
    global_trust: Dict = field(default_factory=dict)

    def __str__(self) -> str:
        return f'num_samples: {self.num_samples},' \
               f' validation_loss: {self.validation_loss},' \
               f' validation_accuracy: {self.validation_accuracy}'
