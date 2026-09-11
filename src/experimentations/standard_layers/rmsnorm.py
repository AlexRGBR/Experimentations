import torch
import torch.nn as nn


class RMSNorm_r(nn.Module):

    def __init__(self, dimensions, eps: float) -> None:
        super().__init__()
        self.dimensions = dimensions
        self.eps = eps

    def _compute_rms(self, x: torch.tensor, N:int):
        variance = torch.sum(torch.square(x))/N
        return torch.sqrt(variance+self.eps)

    def forward(self, input_tensor: torch.tensor):
        output_tensor = input_tensor
        ndim = input_tensor.ndim
        for d in self.dimensions:
            for i in range(d):
                slicer = [slice(None)] * ndim
                slicer[d] = i
                rms = self._compute_rms(output_tensor.select(d, i), d)
                output_tensor[slicer] = output_tensor[slicer] * rms

        return output_tensor