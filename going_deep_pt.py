#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May  7 23:42:54 2022

https://blog.paperspace.com/pytorch-101-advanced/
"""
#%% 

import torch
import torch.nn as nn

#%% Parameter classes 

class net(nn.Module):
  def __init__(self):
    super().__init__()
    self.conv = nn.Linear(10,5)
    
  def forward(self, x):
    return self.linear(x)


myNet = net()

#prints the weights and bias of Linear Layer
print(list(myNet.parameters()))   

#%% not showing other objects

class net1(nn.Module):
  def __init__(self):
    super().__init__()
    self.conv = nn.Linear(10,5)
    self.tens = torch.ones(3,4)                       # This won't show up in a parameter list 
    
  def forward(self, x):
    return self.linear(x)

myNet = net1()
print(list(myNet.parameters()))

#%% TO show up we require to define them as nn.Parameters

class net2(nn.Module):
  def __init__(self):
    super().__init__()
    self.conv = nn.Linear(10,5) 
    self.tens = nn.Parameter(torch.ones(3,4))                       # This will show up in a parameter list 
    
  def forward(self, x):
    return self.linear(x)

myNet = net2()
print(list(myNet.parameters()))

#%% Nesting inside the parameters

class net3(nn.Module):
  def __init__(self):
    super().__init__()
    self.conv = nn.Linear(10,5) 
    self.net  = net2()                      # Parameters of net2 will show up in list of parameters of net3
    
  def forward(self, x):
    return self.linear(x)

myNet = net3()
print(list(myNet.parameters()))

#%% nn.ModuleList to use nn.Parameters

layer_list = [nn.Conv2d(5,5,3), nn.BatchNorm2d(5), nn.Linear(5,2)]

class myNet(nn.Module):
  def __init__(self):
    super().__init__()
    self.layers = layer_list
  
  def forward(self, x):
    for layer in self.layers:
      x = layer(x)

net = myNet()

print(list(net.parameters()))  # Parameters of modules in the layer_list don't show up.

#%% We need modulelist to enlist the parameters of external list (wrapping list using modulelist)

layer_list = [nn.Conv2d(5,5,3), nn.BatchNorm2d(5), nn.Linear(5,2)]

class myNet(nn.Module):
  def __init__(self):
    super().__init__()
    self.layers = nn.ModuleList(layer_list)
  
  def forward(self, x):
    for layer in self.layers:
      x = layer(x)

net = myNet()

print(list(net.parameters()))  # Parameters of modules in layer_list show up.

#%% Weight initialization using the module classes

import matplotlib.pyplot as plt

class myNet(nn.Module):
 
  def __init__(self):
    super().__init__()
    self.conv = nn.Conv2d(10,10,3)
    self.bn = nn.BatchNorm2d(10)
  
  def weights_init(self):
    for module in self.modules():
      if isinstance(module, nn.Conv2d):
        nn.init.normal_(module.weight, mean = 0, std = 1)
        nn.init.constant_(module.bias, 0)

Net = myNet()
Net.weights_init()

for module in Net.modules():
  if isinstance(module, nn.Conv2d):
    weights = module.weight
    weights = weights.reshape(-1).detach().cpu().numpy()
    print(module.bias)                                       # Bias to zero
    plt.hist(weights)
    plt.show()
    
#%% Module vs Children 

class myNet(nn.Module):
  def __init__(self):
    super().__init__()
    self.convBN =  nn.Sequential(nn.Conv2d(10,10,3), nn.BatchNorm2d(10))
    self.linear =  nn.Linear(10,2)
    
  def forward(self, x):
    pass
  

Net = myNet()

print("Printing children\n------------------------------")
print(list(Net.children()))
print("\n\nPrinting Modules\n------------------------------")
print(list(Net.modules()))

#%% Named parameters

for x in Net.named_modules():
  print(x[1], "\n-------------------------------")
  
#%% Learning rate controller

class myNet(nn.Module):
  def __init__(self):
    super().__init__()
    self.fc1 = nn.Linear(10,5)
    self.fc2 = nn.Linear(5,2)
    
  def forward(self, x):
    return self.fc2(self.fc1(x))

Net = myNet()
optimiser = torch.optim.SGD(Net.parameters(), lr = 0.5)


optimiser = torch.optim.SGD([{"params": Net.fc1.parameters(), 'lr' : 0.001, "momentum" : 0.99},
                             {"params": Net.fc2.parameters()}], lr = 0.01, momentum = 0.9)


#%% Learning rate Scheduling and model saving (check the link)

