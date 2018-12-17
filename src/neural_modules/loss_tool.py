import torch
import torch.nn.functional as F
import torch.nn as nn


def log_value_recover(out, index):
    f_out = F.softmax(out, dim=1)   # b, 3
    exp_sum = torch.sum(torch.exp(out), dim=1) # b
    prob_value = 1 - f_out[:, index]    # b
    x = torch.log(prob_value * exp_sum) # b
    return x