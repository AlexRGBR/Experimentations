import numpy as np
import torch
from torch.nn import nn


class RecurrentDeltaNet(nn.module):

    def __init__(self, hidden_dimension, model_dimension):
        super().__init__()

        self.hidden_dimension = hidden_dimension
        self.model_dimension = model_dimension

        self.Q = nn.Linear(model_dimension, hidden_dimension, bias=False)
        self.K = nn.Linear(model_dimension, hidden_dimension, bias=False)
        self.V = nn.Linear(hidden_dimension, hidden_dimension, bias=False)

    def forward(self, x, current_mem):
        q_t = self.Q(x)
        k_t = self.K(x)
        v_t = self.V(x)

        
        mem_update = torch.outer(k_t, v_t)
        new_mem = current_mem + mem_update
        output = q_t @ new_mem
        return output, new_mem


