#%% Library import
from __future__ import  print_function
import  torch

import numpy as np
#%% Basic commands and operation 

x = torch.empty(5,3)
print(x)

x = torch.randn(5, 3 )
print(x)

x =  torch.zeros(5,3, dtype = torch.long)
print(x)


x = torch.tensor([5.6, 3]) 
print(x)


x =  x.new_ones(5, 3 , dtype= torch.double) # if x is already a defined tensor
print(x)


x = torch.randn_like(x, dtype =  torch.float)
print(x)

print(x.size()) # print(x.shape)  gives the same result 


y = torch.rand(5,3)
print(x+y) # print(torch.add(x,y)) or # y.add_(x)

result =  torch.empty(5,3)
torch.add(x,y, out=result)
print(result)

print(x[:,1]) # numpy indexing 


x = torch.randn(4,4)
y = x.view(16)   # kinda reshape
z = x.view(-1, 8 )

print(x.shape, y.shape, z.shape)

x = torch.randn(1)
print(x)
print(x.item()) # only scaler


a = torch.ones(5)
print(a)

b =  a.numpy()
print(b)

# tensor to numpy
a.add_(1)
print(a)
print(b)


# other way around
a = np.ones(5)
b = torch.from_numpy(a)
np.add(a, 1, out=a)
print(a)
print(b)
#%%  GPU
if torch.cuda.is_available():
    device = torch.device("cuda")          
    y = torch.ones_like(x, device=device)  
    x = x.to(device)                      
    z = x + y
    print(z)
    print(z.to("cpu", torch.double)) 
    
#%%  Autograd
    

x = torch.ones(2,2, requires_grad = True)
print(x)    


y = x +2
print(y)    

print(y.grad_fn)


z = y*y*3
out = z.mean()

print(z, out)


a = torch.randn(2, 2)
a = ((a * 3) / (a - 1))
print(a.requires_grad)
a.requires_grad_(True)
print(a.requires_grad)
b = (a * a).sum()
print(b.grad_fn)

# gradient

out.backward()
print(x.grad)


x =  torch.randn(3, requires_grad=True)
y = x*2
while y.data.norm()<1000:
    y = y*2

print(y)


v = torch.tensor([.1, 1 , .0001], dtype =  torch.float)
y.backward(v)

print(x.grad)


print(x.requires_grad)
print((x ** 2).requires_grad)

with torch.no_grad():
    print((x ** 2).requires_grad)
    
# detach for new tensor

print(x.requires_grad)
y = x.detach()
print(y.requires_grad)
print(x.eq(y).all())
