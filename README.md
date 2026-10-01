# Metal Surface Defect Classification using CNN

A deep learning project for **automatic classification of metal surface defects from grayscale images** using a Convolutional Neural Network (CNN) built with TensorFlow/Keras.

The project uses image preprocessing, data augmentation, batch normalization, convolutional feature extraction, global average pooling, and softmax classification to identify six different types of metal surface defects.

---

## 📌 Project Overview

Manual inspection of metal surfaces for defects can be time-consuming and subjective. This project explores a computer-vision-based approach in which a CNN learns visual patterns associated with different surface defects and predicts the defect class of a new image.

### Defect Classes

The model classifies images into six categories:

1. **Crazing**
2. **Inclusion**
3. **Patches**
4. **Pitted**
5. **Rolled**
6. **Scratches**

---

## 🗂️ Dataset

The notebook uses the **Metal Surface Defects Data** dataset in a Kaggle environment.

The expected directory structure is:

```text
Metal Surface Defects Data/
├── train/
│   ├── Crazing/
│   ├── Inclusion/
│   ├── Patches/
│   ├── Pitted/
│   ├── Rolled/
│   └── Scratches/
└── test/
    ├── Crazing/
    ├── Inclusion/
    ├── Patches/
    ├── Pitted/
    ├── Rolled/
    └── Scratches/
```

In the notebook, the dataset is accessed from:

```python
/kaggle/input/metal-surface-defects-data/Metal Surface Defects Data
```

### Dataset split used in the notebook

The notebook creates the training/validation split from the training directory using an **80/20 validation split**.

The generator reported:

- **1,326 training images**
- **330 validation images**
- **72 test images**
- **6 classes**

The test set contains 12 images per class.

---

## 🔬 Methodology

The overall workflow is:

```text
Metal Surface Images
        │
        ▼
Image Loading
        │
        ▼
Resize to 200 × 200
        │
        ▼
Convert to Grayscale
        │
        ▼
Normalize Pixel Values
        │
        ▼
Data Augmentation
        │
        ▼
CNN Feature Extraction
        │
        ▼
Global Average Pooling
        │
        ▼
Dropout
        │
        ▼
Softmax Classification
        │
        ▼
Predicted Defect Class
```

---

## ⚙️ Image Preprocessing

Images are processed using Keras' `ImageDataGenerator`.

### Image parameters

```python
IMG_HEIGHT = 200
IMG_WIDTH = 200
BATCH_SIZE = 16
```

Images are converted to:

```text
200 × 200 × 1
```

where the final dimension represents the single grayscale channel.

### Normalization

Pixel values are rescaled using:

```python
rescale=1./255
```

This converts the original pixel range from:

```text
0–255
```

to:

```text
0–1
```

### Data augmentation

The training pipeline applies:

- Rotation: up to 20°
- Width shift: 10%
- Height shift: 10%
- Shearing: 10%
- Zoom: 10%
- Horizontal flipping
- Nearest-neighbor filling

A 20% validation split is taken from the training directory.

The validation and test images are otherwise normalized without the training augmentation operations.

---

## 🧠 CNN Architecture

The project uses a sequential CNN consisting of three convolutional blocks followed by global average pooling and a six-class output layer.

```text
Input: 200 × 200 × 1

        │
        ▼
Conv2D: 64 filters, 3×3, ReLU
        │
Batch Normalization
        │
Max Pooling: 2×2
        │
        ▼
Conv2D: 128 filters, 3×3, ReLU
        │
Batch Normalization
        │
Max Pooling: 2×2
        │
        ▼
Conv2D: 256 filters, 3×3, ReLU
        │
Batch Normalization
        │
Max Pooling: 2×2
        │
        ▼
Global Average Pooling 2D
        │
        ▼
Dropout: 0.5
        │
        ▼
Dense: 6 neurons, Softmax
```

### Why Global Average Pooling?

The notebook uses `GlobalAveragePooling2D()` instead of flattening the complete feature maps before the dense layer.

This reduces the number of parameters and provides a more compact transition from convolutional feature maps to classification.

### Regularization

A dropout rate of **0.5** is applied before the final classification layer.

---

## 🏋️ Training

The model is compiled using:

```python
Adam(learning_rate=0.001)
```

with:

```python
loss = categorical_crossentropy
metric = accuracy
```

The notebook trains the model for:

```text
60 epochs
```

using the augmented training generator and validation generator.

The notebook was run using a Kaggle environment with an NVIDIA Tesla T4 GPU.

---

## 📊 Results

The notebook reports the following evaluation results.

### Test-set evaluation

The direct `model.evaluate(test_gen)` call reported approximately:

```text
Loss:     0.5082
Accuracy: 80.56%
```

The subsequent classification report gives:

```text
Overall accuracy: 81%
Macro F1-score:   0.78
Weighted F1-score: 0.78
```

The notebook also reports a final validation accuracy of:

```text
89.09%
```

and a final training accuracy of:

```text
98.64%
```

> **Note:** These values come directly from different evaluation outputs in the notebook. The README intentionally preserves the reported results rather than treating them as a single reconciled benchmark.

### Classification Report

| Defect Class | Precision | Recall | F1-Score | Support |
|---|---:|---:|---:|---:|
| Crazing | 1.00 | 1.00 | 1.00 | 12 |
| Inclusion | 0.86 | 1.00 | 0.92 | 12 |
| Patches | 0.55 | 1.00 | 0.71 | 12 |
| Pitted | 1.00 | 0.25 | 0.40 | 12 |
| Rolled | 0.86 | 1.00 | 0.92 | 12 |
| Scratches | 1.00 | 0.58 | 0.74 | 12 |

### Confusion Matrix

The notebook reports the following confusion matrix, with rows representing the true classes and columns representing predicted classes:

```text
[[12,  0,  0,  0,  0,  0],
 [ 0, 12,  0,  0,  0,  0],
 [ 0,  0, 12,  0,  0,  0],
 [ 0,  2,  5,  3,  2,  0],
 [ 0,  0,  0,  0, 12,  0],
 [ 0,  0,  5,  0,  0,  7]]
```

The largest classification difficulties in this evaluation occur for **Pitted** and **Scratches**, with several Pitted samples being classified as other defect types and some Scratches being classified as Patches.

---

## 🧪 Single-Image Prediction

The notebook also demonstrates inference on an individual image.

The input image is:

1. Loaded using PIL
2. Converted to grayscale
3. Resized to `200 × 200`
4. Normalized by dividing by `255`
5. Reshaped to `(1, 200, 200, 1)`
6. Passed through the trained CNN

The predicted class is obtained using:

```python
index = np.argmax(pred[0])
```

and mapped back to the defect name using the generator's class indices.

For the example included in the notebook, the predicted class was:

```text
Crazing
```

---

## 🛠️ Technologies Used

- **Python**
- **TensorFlow / Keras**
- **NumPy**
- **Pandas**
- **Matplotlib**
- **scikit-learn**
- **PIL / Pillow**
- **Jupyter Notebook**
- **Kaggle GPU environment**

---

## 📁 Project Structure

A recommended repository structure is:

```text
metal-surface-defect-classification/
│
├── metal_surface_analysis.ipynb
├── README.md
│
├── images/
│   └── training_history_plots.png
│
└── models/
    └── trained_model.keras
```

The current project submission primarily contains the Jupyter notebook. Model files and generated plots can be added separately if required.

---

## 📈 Evaluation and Visualization

The notebook generates:

- Sample training-image visualizations
- Training/validation accuracy curves
- Training/validation loss curves
- Classification report
- Confusion matrix
- Single-image prediction results

The training history plot is saved as:

```text
training_history_plots.png
```

---

## 🔍 Key Takeaways

- The project demonstrates an end-to-end CNN pipeline for automated metal surface defect classification.
- Grayscale images are used to reduce the input representation to a single channel.
- Data augmentation is used to increase variation in the training data.
- Batch normalization and dropout are incorporated into the CNN.
- Global average pooling is used instead of a large flatten-and-dense block.
- The model successfully distinguishes several defect categories, while the reported confusion matrix shows that **Pitted** and **Scratches** remain comparatively difficult classes for this model/evaluation set.

---
