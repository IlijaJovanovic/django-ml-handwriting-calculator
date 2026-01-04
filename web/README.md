
# Django Inference Application

This directory contains the **Django-based web application** responsible for online inference and user interaction.

The application loads a pre-trained CNN model and performs handwritten expression recognition in real time.

---

## 📁 Directory Structure

```web/
├── ml/ # Inference pipeline
├── ml_models/ # Trained CNN models
├── handcalc/ # Django app
├── templates/ # HTML templates
└── manage.py
```

---

## ▶️ Running the Application

From the `web/` directory:

```bash
python manage.py runserver
```
Then open:

```http://127.0.0.1:8000/```

## Inference pipeline

The inference pipeline performs the following steps:

- Image preprocessing and segmentation
- CNN-based symbol prediction
- Expression reconstruction
- Expression evaluation
- Visual rendering of results

The pipeline logic is orchestrated in:
```
web/ml/pipeline.py
```

## Model loading

The application loads the trained model from:

```
web/ml_models/math_cnn_v1.keras
```

## Notes

- This application is inference-only
- Model updates are handled externally via the training pipeline
- The architecture supports model versioning without code changes
