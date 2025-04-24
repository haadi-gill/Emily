# Emily

1) Collection of data - Given
2) Data Cleaning/Feature Engineering - Try nothing, grayscale, and noise introduction of pixels, Draw bounding box?, Need to encode
3) Model building - CNN
4) Evalute Model - Should have target value
5) Model Deployment - Submission

Data Splitting: Cross-fold stratification (especially with small amount of data)

Preprocessing Ideas:

Labels image into a csv

Make images into GrayScale


Task Splitting:

Script for Grayscale - Alex

Python file with general idea of CNN - Ben, Haadi

Add other code references


# AC Model: Image Classification with ResNetConvNet128

This repository implements a custom image classification pipeline using a modified ResNet-18 backbone and a custom classifier head. The system is designed for multi-class image classification tasks with 12 output classes.

---

## 📂 Components

### 📁 `CustomImageDataset` Class

A subclass of `torch.utils.data.Dataset` that:

- Loads image paths and labels into a single internal DataFrame.
- Applies the specified data augmentations and transformations.
- Prepares image-label pairs for use in PyTorch `DataLoader`.

---

### 🧠 `ResNetConvNet128` Model

A custom deep learning model built on top of a pretrained ResNet-18 backbone with a modified fully connected classifier head.

---

## 🏗️ Model Architecture

### 1. **Backbone: ResNet-18**
- Uses pretrained weights (`weights='DEFAULT'`).
- Removes the final fully connected layer and average pooling.
- Outputs deep feature maps with shape `[B, 512, H, W]`.

### 2. **Adaptive Average Pooling**
- `nn.AdaptiveAvgPool2d((1, 1))` compresses the spatial dimensions.
- Output shape becomes `[B, 512, 1, 1]`.

### 3. **Flattening**
- Output is flattened to `[B, 512]` for the classifier.

### 4. **Classifier Head**
A fully connected classification head with dropout:
- `fc1`: Linear(512 → 256) + ReLU + Dropout
- `fc2`: Linear(256 → 64) + ReLU + Dropout
- `fc3`: Linear(64 → num_classes)

### 5. **Forward Pass**

```python
x = self.features(x)         # ResNet18 feature extractor → [B, 512, H, W]
x = self.pool(x)             # Adaptive pooling → [B, 512, 1, 1]
x = torch.flatten(x, 1)      # Flatten → [B, 512]
x = self.dropout(F.relu(self.fc1(x)))
x = self.dropout(F.relu(self.fc2(x)))
x = self.fc3(x)              # Output logits → [B, 12]
```

## 🧪 Data Preparation & Augmentation

This section describes how input data is preprocessed, augmented, and loaded for model training and validation.

### 🔁 Data Transformations (for Training)

To enhance the generalization ability of the model and reduce overfitting, a series of data augmentation techniques are applied during training:

- **RandomHorizontalFlip**: Randomly flips the image horizontally.
- **RandomRotation(15)**: Rotates the image by up to ±15 degrees.
- **RandomResizedCrop(128, scale=(0.8, 1.0))**: Crops a random region and resizes to 128×128.
- **ColorJitter**: Applies random changes in brightness, contrast, and saturation.
- **ToTensor**: Converts the image from PIL format to PyTorch tensor format.
- **Normalize(mean=[0.5]*3, std=[0.5]*3)**: Normalizes pixel values to the range [-1, 1].

**Transform code:**

```python
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.RandomResizedCrop(128, scale=(0.8, 1.0)),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
])
```
---

### 🔀 Train/Validation Split

The dataset is divided into a training set and a validation set to enable performance monitoring and prevent overfitting during training.

**Details:**
- **Training Set**: Used to train the model parameters.
- **Validation Set**: Used to evaluate performance after each epoch and tune hyperparameters.
- **Split Size**: 750 samples reserved for validation; the rest used for training.

**Code:**
```python
val_size = 750
train_size = len(dataset) - val_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
```

---

### 🚚 Dataloaders

PyTorch `DataLoader`s are used to efficiently handle batching, shuffling, and feeding data into the model during both training and validation phases.

**🔹 Batch Size**  
For training, a batch size of 32 is used, which allows for efficient memory usage and faster convergence

**🔹 Epochs**  
The model is trained for 50 epochs

```python
# DataLoader for batching
n_batch = 32  # Batch size for training
train_loader = DataLoader(train_dataset, batch_size=n_batch, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=val_size, shuffle=False)
```