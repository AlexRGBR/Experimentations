import torch

size = [256, 32]
tens = torch.randn(size)
tens = tens.to(dtype=torch.float16).to('cuda')

max_weights = tens.max(dim=1, keepdim=True)

scale=45
seq_size = 32
a_size = [seq_size, 256]
activations = scale*torch.randn(a_size)

print()