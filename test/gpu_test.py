import torch
print(torch.__version__)
torch.device("mps")
print(torch.cuda.is_available())
torch.cuda.device_count()