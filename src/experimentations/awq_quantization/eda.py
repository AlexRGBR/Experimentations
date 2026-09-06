import torch

size = [256, 32]
tens = torch.randn(size)
# tens = tens.to(dtype=torch.float16).to('cuda')

max_weights = tens.max(dim=1, keepdim=True)

num_activations = 1000
scales=torch.rand(num_activations)*40
seq_size = 32
a_size = [num_activations, seq_size, 256]

activations = torch.randn(a_size)

for i in range(num_activations):
    activations[i,:,:] = scales[i]*activations[i,:,:]

print(activations.shape)
mean_activations = activations.mean(dim=1).mean(dim=0)
# mean_activations = activations.mean(dim=1)


print(mean_activations.shape)

print()