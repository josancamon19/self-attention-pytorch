import torch.nn as nn
import torch
from torch.nn.parameter import Parameter


class SelfAttention(torch.nn):
    def __init__(self, batch_size, embed_dim, head_count):
        
        self.W = Parameter(nn.init.xavier_uniform(torch.zeros((batch_size, ))))
        self.K = Parameter()
        self.V = Parameter()
