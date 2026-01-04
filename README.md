# Handwritten Math Expression Recognition  
### Django Web Application with CNN-based Inference

This project implements a complete **end-to-end machine learning system** for recognizing and evaluating **handwritten mathematical expressions** composed of digits and arithmetic operators (`+`, `-`, `*`).

The system is intentionally designed with a **clear separation between model training and inference**, following real-world machine learning deployment practices.

This repository demonstrates a production-oriented approach to machine learning systems, covering dataset construction, convolutional neural network training, Django-based inference, and reproducible containerized deployment using Docker.

The application follows a multi-stage processing pipeline, including handwritten symbol segmentation, CNN-based classification, expression reconstruction, and evaluation. The system is designed to be portable, reproducible, and ready for deployment in cloud environments.

---

##  Features

- Handwritten digit recognition using the MNIST dataset
- Custom handwritten arithmetic operators (`+`, `-`, `*`)
- Image segmentation of handwritten expressions
- Convolutional Neural Network (CNN) for symbol classification
- Expression reconstruction and evaluation
- Step-by-step visual explanation of the ML pipeline
- Modular and extensible architecture

---

##  System Overview

- User Drawing (Canvas)
  ↓
- Image Segmentation
  ↓
- CNN Inference (Digits & Operators)
  ↓
- Expression Reconstruction
  ↓
- Expression Evaluation
  ↓
- Visual Result


---

## 📁 Project Structure

```puit_projekat/
├── training/ # Offline ML pipeline (datasets, training)
├── web/ # Django inference application
└── README.md 
```


---

## 🛠 Technologies

- Python 3.12
- TensorFlow / Keras
- NumPy
- scikit-learn
- OpenCV
- Pillow
- Django

---

##  Design Decisions

- The division operator (`/`) is intentionally excluded due to visual ambiguity with digit `1`
- Model training is performed strictly offline
- The Django application performs inference only
- Training and inference are fully decoupled

---

##  Future Improvements

- Class weighting for imbalanced datasets
- Parentheses support
- Model versioning and rollback
- Dockerized inference service
- Public REST API

---

##  Author - Ilija Jovanović

This project was developed as a full ML pipeline combining **data engineering, deep learning, and web deployment**, with a focus on clean architecture and real-world applicability.


