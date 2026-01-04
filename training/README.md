# Training Pipeline

This directory contains the **offline machine learning pipeline** used for dataset creation, model training, and model export.

The training pipeline is completely independent from the Django web application and can be executed separately.

---

## 📁 Directory Structure

```training/
├── dataset_pipeline/ # Dataset creation scripts
├── datasets/ # Raw datasets
│ ├── mnist/
│ └── symbols/
│ ├── plus/
│ ├── minus/
│ └── multiply/
├── model/ # CNN architecture and training code
├── export/ # Model export utilities
└── experiments/ # Previous experiments and prototypes
```

---

##  Dataset

### Digits
- MNIST dataset
- 70,000 grayscale images
- Resolution: 28×28

### Operators
Custom handwritten symbols:
- `+` approximately 13,000 samples
- `-` approximately 13,000 samples
- `*` approximately 22,000 samples

All images are:
- converted to grayscale
- resized to 28×28
- normalized to the range `[0, 1]`

---

##  Dataset Generation

To generate the merged dataset:

```bash
python training/dataset_pipeline/merge_datasets.py
```

##  Model Architecture

The CNN architecture is defined in:
```bash
python training/model/train.py
```

This script:

- loads the merged dataset
- splits data into training and validation sets
- trains the CNN
- saves the trained model directly into the inference environment

##  Model Output

Trained models are saved to:
```bash
web/ml_models/math_cnn_v1.keras
```

##  Notes

- Training code is never imported or executed by Django
- Retraining the model does not require changes to the web application
- Dataset imbalance is handled at the training level