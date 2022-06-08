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


#%% Intermediate layers output

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

#%% Looking into gradient [It does so many things in behind]

print(loss.grad_fn)  # MSELoss
print(loss.grad_fn.next_functions[0][0])  # Linear
print(loss.grad_fn.next_functions[0][0].next_functions[0][0])  # ReLU


net.zero_grad()     # zeroes the gradient buffers of all parameters

print('conv1.bias.grad before backward')

# net.parameters # name of the layers and parameter.
print(net.conv1.bias.grad)  # naming interesting like 
# net.conv1.weight.grad

loss.backward()

print('conv1.bias.grad after backward')
print(net.conv1.bias.grad)

# custom update 

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

# torch and torchvision are libraries for python

import torch
import torchvision
import torchvision.transforms as transforms

#%%  One of the most important part is to prepare dataset 

# things get interesting from here

# there are many ways to create the testloader

# link: https://stackoverflow.com/questions/44429199/how-to-load-a-list-of-numpy-arrays-to-pytorch-dataset-loader

# detail: https://pytorch.org/docs/stable/data.html

# way 1: TensorDataset

from PIL import Image
from torch.utils.data import TensorDataset, DataLoader, Dataset

my_x = [np.array([[1.0,2],[3,4]]),np.array([[5.,6],[7,8]])] # a list of numpy arrays
my_y = [np.array([4.]), np.array([2.])] # another list of numpy arrays (targets)
# loop randomly get 16 number 
my_x = np.array(my_x)
my_y = np.array(my_y)

tensor_x = torch.Tensor(my_x)
tensor_y = torch.Tensor(my_y)

my_data1 = TensorDataset(tensor_x,tensor_y)
my_dataload1 = DataLoader(my_data1)

# Now to recall them

dati   = iter(my_dataload1)
im, lb = dati.next()

# or

for i,l in my_dataload1:
    print(i)
    
# Way 2: This will allow us to implement transform 

# we will avoid TensorDataset here

# create new dataset class

# will send list as data and target


class my_dataset(Dataset):
    def __init__(self, data, targets, transform=None):
        self.data = data
        self.targets = torch.LongTensor(targets)
        self.transform = transform
        
    def __getitem__(self, index):
        x = self.data[index]
        y = self.targets[index]
        if self.transform:
            x =  Image.fromarray(self.data[index].astype(np.uint8).transpose(1,2,0))
            x = self.transform(x)
        return x,y
        
    def __len__(self):
        return len(self.data)



my_x = [np.array([[5,2],[6,4]]),np.array([[8.,3],[6,4]])] # a list of numpy arrays

my_y = [np.array([3.]), np.array([1.])] # another list of numpy arrays (targets)

my_x = np.array(my_x)

my_y = np.array(my_y)

cls_data = my_dataset(list(my_x), list(my_y))

my_dataload2 = DataLoader(cls_data)


for i,l in my_dataload2:
    print(i,l)


# we can concatenate the dataset too

d_conc =  torch.utils.data.ConcatDataset([my_data1, cls_data])    

my_dataload3 = DataLoader(d_conc, batch_size=2)


for i,l in my_dataload3:
    print(i,l)

for i, im in enumerate(my_dataload3):
    print(i, im[0], im[1])


# if we understand this then the pytorch data preparation is complete:)


#%% Using Cifar dataset

transform = transforms.Compose(
    [transforms.ToTensor(),   
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)

testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                         shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

#%%

import matplotlib.pyplot as plt
import numpy as np

# functions to show an image


def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


# get some random training images 

# two step processes, iterate the data and load it ... 
    
dataiter = iter(trainloader)
images, labels = dataiter.next()

# images.shape # check it out

# show images
imshow(torchvision.utils.make_grid(images))
# print labels
print(' '.join('%5s' % classes[labels[j]] for j in range(4)))

#%% Net definition



class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
    
net = Net()

#%% loss definition

import torch.optim as optim

criterio = nn.CrossEntropyLoss()

optimizer =  optim.SGD(net.parameters(), lr = 0.001, momentum=0.9)

# Partial parameters update
# the following two lines are dependent to each other. one define and second creats the paramers
# start

partial_param1 = [{'params':net.conv1.parameters()},
                 {'params':net.conv2.parameters()},
                 {'params':net.fc1.parameters()}]

part_optim1 = optim.SGD(partial_param1, lr = 0.001, momentum=0.9)
# end

partial_param2 = [{'params':net.fc1.parameters()},
                  {'params':net.fc2.parameters()}]

part_optim2 = optim.SGD(partial_param2, lr = 0.001, momentum=0.9)

# alternatives: https://discuss.pytorch.org/t/how-the-pytorch-freeze-network-in-some-layers-only-the-rest-of-the-training/7088/15

#%% Network training

net.to(device)

for epoch in range(2):
    running_loss = 0 
    for i, data  in enumerate(trainloader, 0):
        ## we discussed dataloader earlier
        inputs, labels = data[0].to(device), data[1].to(device)
        ## set zero gradient
        ## optimizer.zero_grad()
        part_optim1.zero_grad()
        part_optim2.zero_grad()
        ## output
        outputs = net(inputs)
        ## loss calculation
        loss = criterio(outputs, labels)
        
        # apply gradient
        loss.backward()
        # optimizer.step()
        
        part_optim1.step()
        part_optim2.step()
        
        running_loss += loss.item()
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0    
            

#%% Saving models

PATH = './cifar_net.pth'
torch.save(net.state_dict(), PATH)
type(net.state_dict())
net.state_dict().keys()
# net.state_dict() 
# are dictionary check the keys and values for the ordered dictionary


#%% test the model

dataiter =  iter(testloader)
images, labels = dataiter.next()

# print images
imshow(torchvision.utils.make_grid(images))
print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(4)))


#%% load net and test

net = Net()
net.load_state_dict(torch.load(PATH))

output =  net(images)


_, predicted = torch.max(outputs, 1)

print('Predicted: ', ' '.join('%5s' % classes[predicted[j]]
                              for j in range(4)))

#%% full test set test

correct = 0

total = 0

with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the 10000 test images: %d %%' % (
    100 * correct / total))


#%% class based accuracy


class_correct = list(0. for i in range(10))

class_total = list(0. for i in range(10))
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = net(images)
        _, predicted = torch.max(outputs, 1)
        c = (predicted == labels).squeeze()
        for i in range(4):
            label = labels[i]
            class_correct[label] += c[i].item()
            class_total[label] += 1


for i in range(10):
    print('Accuracy of %5s : %2d %%' % (
        classes[i], 100 * class_correct[i] / class_total[i]))
    
#%% Gpu supprot

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Assuming that we are on a CUDA machine, this should print a CUDA device:

print(device)

#%% 

net.to(device)

inputs, labels = data[0].to(device), data[1].to(device)

#%% Dataset Tutorial 

# https://pytorch.org/tutorials/beginner/basics/data_tutorial.html