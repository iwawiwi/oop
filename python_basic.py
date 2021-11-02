import torch

x = torch.ones(5) # import tensor
y = torch.zeros(3) # expected output
w = torch.randn(5, 3, requires_grad=True) # requires_grad tells pytorch to update this tensor when doing backpropagation
b = torch.randn(3, requires_grad=True)

# initial weight and bias
print(w)
print(b)

z = torch.matmul(x, w) + b
loss = torch.nn.functional.binary_cross_entropy_with_logits(z, y)

print(f"Gradient function for z = {z.grad_fn}")
print(f"Gradient function for loss = {loss.grad_fn}")
print(loss.item())


# computing gradient
loss.backward()
print(w.grad)
print(b.grad)

z = torch.matmul(x, w) + b
print(z.requires_grad)
z = z.detach()
print(z.requires_grad)

with torch.no_grad():
    z = torch.matmul(x, w) + b
print(z.requires_grad)

inp = torch.eye(5, requires_grad=True)
print(f"Initial input grad {inp.grad} and input is {inp}")
out = (inp+1).pow(2)
out.backward(torch.ones_like(inp), retain_graph=True)
print(f"First call {inp.grad}")
out.backward(torch.ones_like(inp), retain_graph=True)
print(f"Second call {inp.grad}")
inp.grad.zero_()
out.backward(torch.ones_like(inp), retain_graph=True)
print(f"Call after zeroing gradients {inp.grad}")