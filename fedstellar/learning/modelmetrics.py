from dataclasses import dataclass, field
from typing import List, OrderedDict, Dict


@dataclass
class ModelMetrics:
    num_samples: int = 0
    validation_loss: float = 0
    cosine_similarity: float = 0
    validation_accuracy: float = 0
    global_trust: Dict = field(default_factory=dict)
