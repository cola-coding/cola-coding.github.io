---
layout: post  
title: "Common Neural Network Layers and APIs in PyTorch"  
date: 2023-03-20  
author: cola  
categories: [Programming, PyTorch]  
image: assets/imgs/cover/Fully_Connected_Layer_Diagram.png  
---

❓Aspiring machine learning beginners often wonder: What exactly happens to an input `tensor` after passing through **multiple neural network operations**？


❓ What are the common methods and `APIs` for transforming a `tensor` **(expanding dimensions, reducing dimensions, reshaping, etc.)**?

❓ Have you ever encountered the following errors but didn't know how to fix them?
```shell
RuntimeError: mat1 and mat2 shapes cannot be multiplied (3584x28 and 12544x512)
RuntimeError: only batches of spatial targets supported (3D tensors) but got targets of dimension: 1
```

**Therefore, this article covers two key points ⬇️**  
- 1. The principles behind common neural network layers  
- 2. Methods and APIs for changing the `shape` of a `tensor`  

---

## 1. Principles of Common Neural Network Layers

The following section provides an overview of how an input `tensor` changes after passing through common neural network layers.

### 1.1 Fully Connected Layer
> The fully connected (FC) operation is a fundamental neural network operation that connects every element of the input `tensor` to every element of the output `tensor`. It applies a **weight matrix** for linear transformation, adds a **bias term**, and produces a new `tensor`. During this process, the `shape` of the `tensor` often changes.

This description might seem abstract, so let's use the diagram below to explain it ⬇️  
<img src="/assets/imgs/ai/PyTorch/model_design/linear-connect.png" width="600" />

Each neuron performs a series of **linear transformations** and then processes the output using an **activation function**. As illustrated above, the fully connected layer links every element in the input `tensor` to every element in the output `tensor`, applies a **weight matrix** for linear transformation, adds a **bias term**, and produces a new `tensor`.

#### Let's look at two 🌰 examples 🌰

For a 2D input `tensor`, if its shape is `(batch_size, input_size)`, then the weight matrix of the fully connected layer typically has a shape of `(input_size, output_size)`, resulting in an output `tensor` of shape `(batch_size, output_size)`.

```python
torch.Size([2, 3]) # Input: batch_size = 2, input_size = 3
nn.Linear(3, 4)  # After fully connected layer: input_size = 3, output_size = 4
torch.Size([2, 4]) # Output: batch_size = 2, output_size = 4
```
---

For a 3D input tensor with shape `(batch_size, channels, input_size)`, where `channels` represents the number of input channels and `input_size` is the number of input features, the fully connected layer's weight matrix is typically of shape `(input_size, output_size)`. The fully connected operation multiplies the input tensor with the weight matrix, adds a bias term, and produces an output tensor of shape `(batch_size, channels, output_size)`.
```python
torch.Size([2, 3, 4]) # Input: batch_size = 2, channels = 3, input_size = 4
nn.Linear(4, 5) # After fully connected layer: input_size = 4, output_size = 5 
torch.Size([2, 3, 5]) # Output: batch_size = 2, channels = 3, output_size = 5
```

### 1.2 Convolutional Layer
> The convolution operation is a fundamental neural network operation that slides a fixed-size window over the input tensor, performing **weighted summation** on the values within the window to generate a new tensor. During this process, the tensor's shape may change depending on the kernel size and stride used.

For example, consider a 5×5 image and a 3×3 convolutional kernel ⬇️  

<img src="/assets/imgs/ai/PyTorch/Model_Design/cnn-01.png" width="600" />

First, apply **weighted summation** to the first window of data. The computed result is 27 ⬇️  

<img src="/assets/imgs/ai/PyTorch/Model_Design/cnn-02.png" width="600" />

❓ Why is the output a 3×3 matrix? Let's illustrate using a **stride of 1**, meaning we move one step at a time ⬇️  

<img src="/assets/imgs/ai/PyTorch/Model_Design/cnn-03.png" width="600" />

The convolution kernel slides over the image, producing an output with width:
> (Original Width - Kernel Width + 2 × Padding (not covered here)) / Stride + 1

The height is calculated similarly.

### 1.3 Pooling Layer

> **Pooling** is a widely used feature **dimensionality reduction** method, commonly found in convolutional neural networks. The pooling operation slides a fixed-size window over the input tensor and applies an **aggregation operation**, such as **max pooling** or **average pooling**, to produce a new tensor.

Pooling operations alter the tensor’s shape, typically reducing spatial dimensions (width and height) while preserving depth (number of channels).

<img src="/assets/imgs/ai/PyTorch/Model_Design/pooling-01.png" width="600" />  
<img src="/assets/imgs/ai/PyTorch/Model_Design/pooling-02.png" width="600" />

> Note that pooling operations do not usually change the batch size (first dimension), so the batch size remains unchanged in the output tensor.

---

## 2. Changing Tensor Dimensions in PyTorch
Additionally, beginners often wonder how to obtain a tensor of a specific shape (for mathematical operations, loss calculations, etc.), such as **expanding, reducing, or reshaping dimensions**. Which APIs should be used to modify a tensor?

### 2.1 `view`
Returns a new tensor with the same data but a different `shape`.  
> Returns a new tensor with the same data as the self tensor but of a different shape.

```python
import torch
x = torch.randn(4, 4)
x.size() # torch.Size([4, 4])
y = x.view(16)
y.size() # torch.Size([16])
z = x.view(-1, 8)  # The size -1 is inferred from other dimensions
z.size() # torch.Size([2, 8])

a = torch.randn(1, 2, 3, 4)
a.size() # torch.Size([1, 2, 3, 4])
b = a.transpose(1, 2)  # Swaps 2nd and 3rd dimensions
b.size() # torch.Size([1, 3, 2, 4])
c = a.view(1, 3, 2, 4)  # Does not change tensor layout in memory
c.size() # torch.Size([1, 3, 2, 4])
```

### 2.2 `unsqueeze`
Returns a new `tensor` with a new dimension inserted at the specified position. For example:

```python
import torch
x = torch.tensor([1, 2, 3, 4])
torch.unsqueeze(x, 0) # tensor([[ 1,  2,  3,  4]])
torch.unsqueeze(x, 1)
# tensor([[ 1],
#         [ 2],
#         [ 3],
#         [ 4]]) 
```

### 2.3 `split`
Splits a `tensor` in place, modifying the original `tensor`. For example:

```python
a = torch.arange(10).reshape(5, 2)
print(a)
# tensor([[0, 1],
#         [2, 3],
#         [4, 5],
#         [6, 7],
#         [8, 9]])
torch.split(a, 2)
# (tensor([[0, 1], [2, 3]]),
#  tensor([[4, 5], [6, 7]]),
#  tensor([[8, 9]]))
```

The above covers the principles of common neural network layers and some APIs for changing tensor shapes.  
For more details, refer to the **PyTorch official documentation**, which provides extensive APIs.