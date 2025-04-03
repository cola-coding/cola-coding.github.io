---
layout: post  
title: "PyTorch Data Transformation"  
date: 2023-03-15  
author: cola  
categories: [Programming, PyTorch]  
image: assets/imgs/cover/data_preprocessing_basics.png  
---

Converting images into `PyTorch Tensors` is a common task in the field of computer vision. This article mainly introduces related data transformations.
 

## 1. Convert Images to PyTorch Tensors  
You can use the `transforms` module in `torchvision`, a tool from `PyTorch`, to convert images into `PyTorch Tensors`.  

<img src="/assets/imgs/ai/PyTorch/data-process-loading/img2tensor.png" width="400" />  

> A `Tensor` includes `scalar`, `vector`, `matrix`, and multi-dimensional tensors.

<img src="/assets/imgs/ai/PyTorch/data-process-loading/tensor-introduce.png" width="400" />  

Below is an example of using the `torchvision.transforms` library to convert an image into a `PyTorch Tensor`:  

```python
import torch
import torchvision.transforms as transforms
from PIL import Image 

# Create a transforms object
transform = transforms.Compose([
    transforms.Resize((224, 224)), # Resize the image to 224x224
    transforms.ToTensor() # Convert the image to a PyTorch tensor
])

# Load the image
img = Image.open('image.jpg')

# Convert the image to a PyTorch tensor
img_tensor = transform(img)

# Check the shape and data type of the tensor
print(img_tensor.shape) # torch.Size([3, 224, 224]) 
print(img_tensor.dtype) # torch.float32
```
The above process can be summarized as follows ⬇️  

- 1. **Define Transform**: Use the `Compose()` function from `torchvision.transforms` to create a `transforms` object, which includes two transformation operations: `Resize()` resizes the image to `224x224`, and `ToTensor()` converts the image to a `PyTorch Tensor`.  

- 2. **Open Image**: Use `Image.open()` from the `PIL` image processing library to load an image named `image.jpg`.  

- 3. **Transform**: Use the `transform()` method to convert the image into a `PyTorch Tensor`.  

- 4. **Check Information**: Use the `shape` and `dtype` attributes to check the shape and data type of the tensor.  

## 2. Normalize Image Data  

Normalizing image data scales pixel values to a fixed range, ensuring that all data is on the same scale. This can improve model training speed and performance while reducing the risk of overfitting.  

### 2.1 Common Normalization Methods  

Some common normalization methods include:  

- **Min-Max Normalization**: Scales pixel values to the [0,1] range. Each pixel value `x` is transformed using the formula `(x - min)/(max - min)`, where `min` and `max` are the minimum and maximum pixel values in the image.  

- **Z-score Normalization**: Scales pixel values to have a mean of 0 and a standard deviation of 1. Each pixel value `x` is transformed using `(x - mean)/std`, where `mean` and `std` are the mean and standard deviation of all pixel values in the image.  

- **Divide by 255**: Scales pixel values to the [0,1] range by dividing each pixel value by 255, which is effectively Min-Max Normalization.  

<img src="/assets/imgs/ai/PyTorch/data-process-loading/normalizations.png" width="400" />  

In `PyTorch`, you can use the `torchvision.transforms.Normalize()` function to normalize image data. 

This function applies `Z-score normalization`, which will be explained in more detail later.  

Here's an example:  

```python
import torch
import torchvision.transforms as transforms

# Define mean and standard deviation
mean = [0.5, 0.5, 0.5]
std = [0.5, 0.5, 0.5]

# Define transformation
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

# Load image data and apply transformation
img = Image.open('example.jpg')
img_transformed = transform(img)
```
In the above code, the `transforms.Normalize()` function standardizes each channel of the input data using the specified mean and standard deviation. 

**If the input data is a three-channel image, the mean and standard deviation should be specified for all three channels.**  

The overall image data preprocessing process is shown below⬇️  

<img src="/assets/imgs/ai/PyTorch/data-process-loading/data-preprocess-steps.png" width="6000" />  

### 2.2 ToTensor vs Normalize  

`torchvision.transforms.ToTensor()` and `torchvision.transforms.Normalize()` are both commonly used image data preprocessing functions in `PyTorch`, but they serve different purposes.  

- **Data Conversion**: The `torchvision.transforms.ToTensor()` function converts images in `PIL.Image` format to `PyTorch Tensor` format. The resulting tensor has a shape of C×H×W, where C represents the number of channels (usually 3 for RGB images or 1 for grayscale images), and H and W represent the image height and width. `It scales pixel values from integer values (0 to 255) to floating-point values (0 to 1).`  

- **Data Normalization**: The `torchvision.transforms.Normalize()` function is used for `data normalization`. It takes two parameters: mean and standard deviation, which are used to standardize the data. Typically, this function is used to normalize training and test data so that they have the same distribution. For image data, the mean is usually set to `[0.5, 0.5, 0.5]` and the standard deviation to `[0.5, 0.5, 0.5]`, scaling pixel values to the `-1 to 1` range.  

When using these functions, you typically apply `torchvision.transforms.ToTensor()` first to convert the image into a `PyTorch Tensor`, followed by `torchvision.transforms.Normalize()` for normalization.  

### 2.3 Normalize Algorithm Principle — Z-score Normalization  

For an input tensor `x`, the `Normalize()` function first subtracts the mean `mean` from each element in `x`, then divides by the standard deviation `std`, i.e., `(x - mean) / std`.  

This operation ensures that each element in `x` follows a standard normal distribution with a mean of 0 and a standard deviation of 1.  

**The normalization operation is performed separately for each channel, so `mean` and `std` should be lists of values for each channel.**  

## 3. Extension: Convert Tensor Data Back to Image  

In `PyTorch`, you can use the `make_grid` function from the `torchvision.utils` module to arrange multiple images into a grid and convert them into a `PIL` image object. 

The following steps outline how to convert `tensor` data back into an image:  

- 1. Convert `tensor` data into image data, such as scaling pixel values from `[0,1]` to `[0,255]` integer values and converting them into a `PIL` image object.  
- 2. Use the `make_grid` function to arrange multiple images into a grid with specified rows, columns, and spacing. The `make_grid` function returns a tensor containing the arranged image data.  
- 3. Convert the returned tensor into a `numpy` array and scale pixel values from `[0,1]` to `[0,255]` integer values.  
- 4. Convert the `numpy` array into a `PIL` image object.  

Below is an example⬇️  

```python
import torch
import torchvision.transforms as transforms
from torchvision.utils import make_grid
from PIL import Image

# Create a random tensor of size 3x256x256
tensor = torch.rand(3, 256, 256)

# Convert tensor data to [0, 255] integer values
tensor = tensor * 255
tensor = tensor.byte()

# Create a transformation function to convert tensor data into a PIL image object
to_pil = transforms.ToPILImage()

# Convert tensor data into a PIL image object
img = to_pil(tensor)

# Arrange multiple images into a grid and convert them into tensor data
grid = make_grid([tensor, tensor, tensor], nrow=3, padding=10)
grid = grid.mul(255).permute(1, 2, 0).byte().numpy()

# Convert tensor data into a PIL image object
img = Image.fromarray(grid)
```

Note that when using `make_grid`, the input should be a list containing the tensors to be arranged.