import random

import numpy as np
import torch


def set_seeds() -> None:
    """
    set the seeds for torch, numpy and random
    """
    torch.manual_seed(12345)
    random.seed(12345)
    np.random.seed(12345)
