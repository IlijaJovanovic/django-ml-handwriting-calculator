from django.shortcuts import render
from django.http import JsonResponse
from PIL import Image
import io
import base64
import json

from ml import predict_expression


def index(request):
    return render(request, "handcalc/index.html")


def predict(request):
    if request.method != "POST":
        return JsonResponse({"error": "Invalid request method"}, status=405)

    try:
        # Parse JSON body
        data = json.loads(request.body)

        # Decode base64 image
        img_data = data["image"].split(",")[1]
        img_bytes = base64.b64decode(img_data)
        image = Image.open(io.BytesIO(img_bytes)).convert("L")

        # Run ML pipeline
        result = predict_expression(image)

        return JsonResponse(result)

    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)
