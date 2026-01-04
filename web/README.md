
# Django Inference Application

This directory contains the **Django-based web application** responsible for online inference and user interaction.

The application loads a pre-trained CNN model and performs handwritten expression recognition in real time.

---

##  Directory Structure

```web/
├── ml/ # Inference pipeline
├── ml_models/ # Trained CNN models
├── handcalc/ # Django app
├── templates/ # HTML templates
└── manage.py
```

---

##  Running the Application

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

#  Docker Deployment

The Django inference application is fully containerized using Docker, enabling reproducible and environment-independent deployment.

The Docker image includes:
- Python runtime and required system dependencies
- All Python packages defined in `requirements.txt`
- The Django application code
- The pre-trained CNN model used for inference

## Build the Docker image

From the project root directory:

```bash
docker build -t handcalc-web ./web
```

Run the application using Docker
```
docker run -p 8000:8000 handcalc-web
```
The application will be available at:
```
http://127.0.0.1:8000/
```
## Notes on Docker usage

- The Docker container runs the application in an isolated environment without requiring a local Python installation.
- Port 8000 is exposed by the container and mapped to the host machine at runtime.
- Any code or dependency changes require rebuilding the Docker image.

## Why Docker?
Docker is used to ensure:

- Consistent behavior across different systems
- Simplified deployment on cloud platforms such as AWS EC2
- Clear separation between development, inference, and runtime environments

