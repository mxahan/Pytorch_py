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
#%%  GPU (check this carefully, for some update issues this may not work next time you run :(
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


#%% Neural network 

import torch 
import torch.nn as nn
import torch.nn.functional as F 


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        
        self.conv1 = nn.Conv2d(1, 6, 3)
        self.conv2 = nn.Conv2d(6, 16, 3)
        
        self.fc1 = nn.Linear(16*6*6, 120)
        self.fc2 = nn.Linear(120, 84)
        
        self.fc3 = nn.Linear(84, 10)
        
    def forward(self, x):
        
        x = F.max_pool2d(F.relu(self.conv1(x)), (2,2))
        
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        
        x = x.view(-1, self.num_flat_features(x))
        
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        
        x = self.fc3(x)
        
        return x
    def num_flat_features(self, x):
        
        size = x.size()[1:]
        
        num_features = 1 
        for s in size:
            num_features= num_features*s
        return num_features
            
    
net = Net()

print(net)

#%% Parametes

params =  list(net.parameters()) # like trainables in tf

# list like params[0].data
print(len(params))
print(params[0].size())


#%%
input = torch.randn(1, 1, 32, 32)
out = net(input)
print(out)

net.zero_grad()
out.backward(torch.randn(1, 10))  # random gradient


#%% Intermediate layers

out1 = net.conv1(input) # to be appropriate do what the forward did 
out1 = F.max_pool2d(F.relu(net.conv1(input)), (2,2))
out2 = F.max_pool2d(F.relu(net.conv2(out1)), (2,2))  # out2.shape

out3 = out2.view(-1, net.num_flat_features(out2))

out4 = F.relu(net.fc1(out3))  # so on and so forth

#%% Loss function

output = net(input)
target = torch.randn(10)  # a dummy target, for example
target = target.view(1, -1)  # make it the same shape as output
criterion = nn.MSELoss()

loss = criterion(output, target)
print(loss)
#%%
print(loss.grad_fn)  # MSELoss
print(loss.grad_fn.next_functions[0][0])  # Linear
print(loss.grad_fn.next_functions[0][0].next_functions[0][0])  # ReLU


net.zero_grad()     # zeroes the gradient buffers of all parameters

print('conv1.bias.grad before backward')
print(net.conv1.bias.grad)  # naming interesting like 
# net.conv1.weight.grad

loss.backward()

print('conv1.bias.grad after backward')
print(net.conv1.bias.grad)


learning_rate = 0.01
for f in net.parameters():
    f.data.sub_(f.grad.data * learning_rate)
    
#or the followings 

#%%

import torch.optim as optim

# create your optimizer
optimizer = optim.SGD(net.parameters(), lr=0.01)

# in your training loop:
optimizer.zero_grad()   # zero the gradient buffers
output = net(input)
loss = criterion(output, target)
loss.backward()
optimizer.step()    # Does the update
    

#%% training a classifier parts
