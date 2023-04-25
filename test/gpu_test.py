import torch
from pytorch_lightning import Trainer

print(torch.__version__)
torch.device("mps")
print(torch.cuda.is_available())
torch.cuda.device_count()

trainer = Trainer(accelerator="mps", devices=1)