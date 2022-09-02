import torch

print(torch.cuda.is_available())
a = torch.rand(5,3)

print(a)