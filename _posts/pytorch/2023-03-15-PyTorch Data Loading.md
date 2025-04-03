---
layout: post
title: "PyTorch Data Loading"
date: 2023-03-15
author: cola
categories: [Programming, PyTorch]
image: assets/imgs/cover/data_loading_sequence.jpeg
---

After understanding some details of **[PyTorch Data Transformation]**, we need to load the data for training. This article introduces PyTorch data preprocessing.

## PyTorch Data Preprocessing
  
`PyTorch` provides built-in `Dataset` and `DataLoader` classes for this purpose.

> The `Dataset` (such as `torch.utils.data.Dataset`, `torchvision.datasets.ImageFolder`, etc.) encapsulates data transformation details. 

> The `DataLoader` (`torch.utils.data.DataLoader`) is then used to load data in parallel.

The following sequence diagram roughly illustrates the data loading process ⬇️:  
<img src="/assets/imgs/ai/PyTorch/data-process-loading/data-loading-sequence.jpeg" width="800" />


## Data Loading ⚙️

`DataLoader` is a class in `PyTorch` used for batch data loading. It accepts any dataset object from `PyTorch` (such as `torch.utils.data.Dataset`, `torchvision.datasets.ImageFolder`), transforming it into a batch-processing and parallel-loading data loader.


<img src="/assets/imgs/ai/PyTorch/data-process-loading/DataLoader-args.jpeg" width="600" />

When using `DataLoader`, we can configure several parameters to control data loading and batching, including:

- **batch_size**: Number of samples per batch.
- **shuffle**: Whether to shuffle the dataset before each epoch.
- **num_workers**: Number of processes used for data loading.
- **drop_last**: Whether to drop the last incomplete batch if the dataset size is not divisible by the batch size.

Here’s a simple example demonstrating how to use `DataLoader` to load a dataset:

```python
import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms

# Create a dataset object
data_path = "/path/to/your/data"
data_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5],
                         std=[0.5, 0.5, 0.5])
])
dataset = datasets.ImageFolder(data_path, transform=data_transform)

# Create a data loader object
data_loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)

# Iterate through the dataset
for images, labels in data_loader:
    # Process each batch of data
    pass
```

1. First, we create a `transforms.Compose` object (see **[PyTorch Data Transformation]**) and an `ImageFolder` dataset object to preprocess and load image data.

2. Then, we use `DataLoader` to create a data loader object, passing in the dataset and additional parameters. Finally, we iterate through the dataset using a `for` loop to process each batch of data.

> Note: When using `DataLoader`, batch size and the number of data loading processes should be adjusted based on hardware capabilities to achieve optimal performance and efficiency.