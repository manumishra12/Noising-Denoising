# Noising and Denoising of Images

Welcome to the **Noising and Denoising of Images** repository! This project demonstrates the application of deep learning techniques to add noise to images and then reconstruct the original images by removing the noise using a denoising autoencoder.

---

## 📋 Table of Contents
- [Introduction](#introduction)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Technical Details](#technical-details)
- [Installation and Usage](#installation-and-usage)
- [Results](#results)
- [Future Work](#future-work)
- [License](#license)

---

## 🖼️ Introduction
The goal of this project is to explore image noising and denoising tasks using a custom PyTorch dataset and a denoising autoencoder. It introduces artificial noise into images and then trains a model to recover the original images, making it a useful application in fields like image restoration, medical imaging, and computer vision preprocessing.

**Key Features:**
1. Custom Dataset class for flexible image loading and noising.
2. Noise types supported: **Gaussian noise** (extendable to others).
3. Denoising Autoencoder using convolutional layers for image reconstruction.

---

## 📂 Dataset
### **NoisyImageDataset**
A custom PyTorch `Dataset` class is used to handle image loading and augmentation:
- **Directory-based Loading**: Recursively loads images from a given directory (`root_dir`).
- **Noise Addition**: Adds noise to images using the `random_noise` function from `skimage`.

#### **Initialization Parameters:**
- `root_dir`: Path to the directory containing images.
- `transform`: Optional PyTorch transforms applied to the images.
- `noise_type`: Type of noise to add (default: `gaussian`).
- `noise_amount`: Intensity of the noise (variance for Gaussian noise).

#### Example Usage:
```python
dataset = NoisyImageDataset(
    root_dir="data/train",
    transform=transform,
    noise_type="gaussian",
    noise_amount=0.1
)
```
## 🧠 Model Architecture
### **Denoising Autoencoder**
This project uses a convolutional autoencoder, a deep learning model designed to encode the input to a lower-dimensional representation and then decode it back to the original dimensions.

#### **Structure:**
1. **Encoder**:
   - Downsamples the image using convolutional layers.
   - Extracts important features for reconstruction.
2. **Decoder**:
   - Reconstructs the original image from the encoded features using transposed convolutions.

#### **Model Summary:**
```python
class DenoisingAutoencoder(nn.Module):
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
```

## 💻 Installation and Usage

### **Prerequisites**
- Python 3.8+
- PyTorch 1.10+ (with CUDA support for GPU acceleration)
- Additional libraries: `torchvision`, `Pillow`, `scikit-image`

### **Installation**
1. Clone the repository:
   ```bash
   git clone https://github.com/username/noising-denoising-images.git
   cd noising-denoising-images
   ```

2. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Organize your dataset such that all images are in a directory (e.g., data/train).
Train the model:

   ```bash
   python train.py
   ```
   
4. Evaluate the model:
   ```bash
   python evaluate.py
   ```

## 📊 Results

The model effectively removes Gaussian noise from images after training. Below is an example of the output:

![Results](https://github.com/manumishra12/Noising-Denoising/blob/main/Images/result2.png)

The image showcases three sections:
1. **Input (Noisy)**: The input image with added Gaussian noise.
2. **Output (Denoised)**: The reconstructed image after denoising by the model.
3. **Ground Truth**: The original, clean image for comparison.

![Results](https://github.com/manumishra12/Noising-Denoising/blob/main/Images/result1.png)

---

## 🚀 Future Work
1. Add support for additional noise types (e.g., salt-and-pepper, speckle noise).
2. Enhance the architecture using advanced models (e.g., U-Net or residual connections).
3. Extend the project to handle video denoising tasks.
4. Explore transfer learning for domain-specific denoising tasks.
