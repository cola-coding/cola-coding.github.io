---
layout: post  
title: "PyTorch Metric Visualization"  
date: 2023-03-17  
author: cola  
categories: [Programming, PyTorch]  
image: assets/imgs/cover/image_loading_folder.png  
---

During the training of neural network models, it is often necessary to visualize output metrics. Here, I recommend two real-time data visualization libraries: `TensorBoardX` and `Visdom`.

**Visdom** is a Python library for real-time data visualization, particularly suitable for the `PyTorch` deep learning framework. It can be used to monitor and visualize model training and evaluation metrics in real time, such as **loss function and accuracy.**  

---

Below is a brief introduction to how to use `Visdom` ⬇️  

## 1. Using Visdom in PyTorch  

The following are the steps to use `Visdom` for visualization in `PyTorch`:  

### 1.1 Install and Start the Visdom Server  

First, install `Visdom` using `pip`:  

```bash
pip install visdom
```
The Visdom startup script is as follows:  

```bash
python -m visdom.server
```

Import `Visdom` in a `Python` script:  

```python
import visdom
vis = visdom.Visdom()
```
---

### 1.2 Writing a Simple Neural Network Training Program  
The neural network model definition and training involve the following basic steps ⬇️  

```markdown
- Data preprocessing and loading model definition
- Model definition
- Loss function and optimizer definition
- Start training
    - Load processed image data
    - Start training (iterations, etc.)
    - Compute loss, etc.
    - Backpropagation to update parameters
```

Here, I have drawn a simple flowchart to illustrate the process ⬇️  
<img src="/assets/imgs/ai/PyTorch/model_design/train-steps.png" width="800" />

The specific code implementation is as follows.  

#### 1.2.1 Create a Dataset Object  
```python
data_path = "./data/dogcat" # Image loading path, each subfolder contains images classified accordingly, which is a prerequisite for using imageFolder
```
<img src="/assets/imgs/ai/PyTorch/model_design/image_loading_folder.png" width="400" />

```python
# Define transform for image preprocessing
data_transform = transforms.Compose([
    transforms.Resize(40), # Resize
    transforms.CenterCrop(32), # Center crop
    transforms.ToTensor(), # Convert to tensor
    transforms.Normalize(mean=[0.5, 0.5, 0.5],
                         std=[0.5, 0.5, 0.5]) # Normalize data
])
dataset = datasets.ImageFolder(data_path, transform=data_transform) # The prerequisite for using imageFolder is that images are already categorized in subfolders
```

#### 1.2.2 Create a Data Loader Object  
Here, `DataLoader` is used for batch loading of data.  

```python
data_loader = torch.utils.data.DataLoader(dataset, batch_size=10, shuffle=True, num_workers=4)
```

#### 1.2.3 Define Model and Optimizer  
A simple neural network model is defined along with `optimizer` and loss function. `PyTorch` provides many built-in templates and methods for us to use (very convenient, haha).  

```python
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.nn.functional as F

# Define the network model
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # First layer (convolutional layer)
        self.conv1 = nn.Conv2d(3,6,3) # Input channels: 3, Output channels: 6, Convolution: 3x3
        # Second layer (convolutional layer)
        self.conv2 = nn.Conv2d(6,16,3) # Input channels: 6, Output channels: 16, Convolution: 3x3
        # Third layer (fully connected layer)
        self.fc1 = nn.Linear(16*28*28, 512) # Input dimension: 16x28x28=12544, Output dimension: 512
        # Fourth layer (fully connected layer)
        self.fc2 = nn.Linear(512, 64) # Input dimension: 512, Output dimension: 64
        # Fifth layer (fully connected layer)
        self.fc3 = nn.Linear(64, 10) # Input dimension: 64, Output dimension: 10

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        
        x = self.conv2(x)
        x = F.relu(x)
        
        x = x.view(-1, 16*28*28)
        x = self.fc1(x)
        x = F.relu(x)
        
        x = self.fc2(x)
        x = F.relu(x)
        
        x = self.fc3(x)
        return x

net = Net()
# Define optimizer and loss function
optimizer = optim.SGD(net.parameters(), lr=0.01)
criterion = nn.CrossEntropyLoss()
```

#### 1.2.4 Define a Function to Visualize Loss Function  

Next, we use `Visdom` to visualize the output. We plot the x and y coordinates and add `title` and other annotations ⬇️  

```python
def plot_loss(loss, epoch):
    vis.line(
        X=[epoch],
        Y=[loss],
        win='loss',
        update='append',
        opts=dict(title='Training Loss', xlabel='Epoch', ylabel='Loss')
    )
```

#### 1.2.5 Start Training  

```python
for epoch in range(10):
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data # Load image data
        optimizer.zero_grad()
        outputs = net(inputs) # Training
        loss = criterion(outputs, labels) # Compute loss
        loss.backward() # Backpropagation
        optimizer.step()
        running_loss += loss.item()
        plot_loss(running_loss / 10, epoch) # Call metric visualization function
        running_loss = 0.0
```
---

## 2. View Results  

```bash
# First, start the Python program
python visdom_test.py

# Start visdom
python -m visdom.server
```

Finally, open http://localhost:8097/ to see the visualized metrics.

These are the basic steps to use `Visdom` for visualization in `PyTorch`. We can also customize it using other available functions and parameters as needed.

## 3. How Does It Work?  

The principle is actually very simple: two processes are started, one for the main training workflow and the other for the Visdom web process. These two processes communicate in real time, allowing the metrics to be displayed on the front-end interface in real time.