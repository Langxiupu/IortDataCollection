from config import TrainConfig
import itertools
import numpy as np
import torch

a = torch.tensor([1, 2, 3])
b = torch.tensor([4, 5, 6])
print(torch.stack((a, b), axis=0))