import math
import pandas as pd
import numpy as np
import torch.nn as nn
from sklearn.metrics import confusion_matrix
from scipy.optimize import linear_sum_assignment



def he_init_weights(module):
    """
    Initialize network weights using the He (Kaiming) initialization strategy.

    :param module: Network module
    :type module: nn.Module
    """
    if isinstance(module,  nn.Linear):
        nn.init.xavier_uniform_(module.weight, 1)




