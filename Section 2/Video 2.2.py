import torch
A = torch.rand(3,3)
x = torch.rand(3)
b = torch.mv(A, x)
print(b)
print(A.shape)
