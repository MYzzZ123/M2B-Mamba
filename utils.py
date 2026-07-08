import random
import numpy as np
import torch

def fix_seeds(seed, with_torch=True, with_cuda=True):
    random.seed(seed)
    np.random.seed(seed)
    if with_torch:
        torch.manual_seed(seed)
    if with_cuda:
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True

def stable_softmax(x, dim=1):
    max_x = torch.max(x, dim=dim, keepdim=True).values
    exp_x = torch.exp(x - max_x)
    return exp_x / torch.sum(exp_x, dim=dim, keepdim=True)

