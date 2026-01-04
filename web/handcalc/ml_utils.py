import numpy as np
from PIL import Image,ImageFilter
import cv2
import os
#from tensorflow.keras.models import load_model
from keras._tf_keras.keras.models import load_model
import random

model = load_model("ml_models/math_cnn.keras")

LABELS = {0: '0', 1: '1', 2: '2', 3: '3', 4: '4',
          5: '5', 6: '6', 7: '7', 8: '8', 9: '9',
          10: '+', 11: '-',12:'*'}

def create_image(number):
    folder_path="brojevi"
    digits = str(number)
    digit_images = []

    for digit in digits:
        variant = random.randint(0, 9)
        filename = f"{digit}_{variant}.png"
        path = os.path.join(folder_path, filename)

        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing file: {path}")

        img = Image.open(path).convert("L")  # Convert to grayscale if needed
        digit_images.append(img)

    # Calculate total width and max height
    widths, heights = zip(*(img.size for img in digit_images))
    total_width = sum(widths)
    max_height = max(heights)

    # Create a new blank image and paste digits side by side
    final_image = Image.new("L", (total_width, max_height), color=255)  # white background

    x_offset = 0
    for img in digit_images:
        final_image.paste(img, (x_offset, 0))
        x_offset += img.width
    final_image.save(f"segmented_debug/spojeno.png")
    return final_image
def model_predict(image):
    img_array = np.array(image).astype("float32") / 255.0
    img_array = img_array.reshape(1, 28, 28, 1)
    prediction = model.predict(img_array)
    print(prediction)
    print(f'{max(prediction[0])}')
    return LABELS[np.argmax(prediction)]

def evaluate_expression(symbols):
    print(symbols)
    expr = ''.join(symbols)
    try:
        return eval(expr)
    except Exception as e:
        return f"{e}"

def zoom_image(image, zoom_factor=0.85, final_size=28):
    h, w = image.shape
    crop_h = int(h * zoom_factor)
    crop_w = int(w * zoom_factor)

    y1 = (h - crop_h) // 2
    x1 = (w - crop_w) // 2
    y2 = y1 + crop_h
    x2 = x1 + crop_w

    zoomed = image[y1:y2, x1:x2]
    zoomed_resized = cv2.resize(zoomed, (final_size, final_size), interpolation=cv2.INTER_AREA)
    return zoomed_resized

def resize_and_pad(image, size=28, pad_value=0):
    h, w = image.shape
    scale = size / max(w, h)
    resized = cv2.resize(image, (int(w * scale), int(h * scale)))
    square = np.full((size, size), pad_value, dtype=np.uint8)
    y_offset = (size - resized.shape[0]) // 2
    x_offset = (size - resized.shape[1]) // 2
    square[y_offset:y_offset + resized.shape[0], x_offset:x_offset + resized.shape[1]] = resized
    return square

def segment_characters(pil_img, debug=True):
    import numpy as np
    from PIL import Image
    import cv2
    import cv2
    import os

    img = np.array(pil_img)

    # Ensure white symbols on black background
    _, thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Optional: improve separation of digits/operators
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    thresh=cv2.bitwise_not(thresh)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = sorted([cv2.boundingRect(c) for c in contours], key=lambda b: b[0])
    print(boxes)
    symbol_images = []

    for i, (x, y, w, h) in enumerate(boxes):
        if w < 10 or h < 10:
            continue

        roi = thresh[y:y+h, x:x+w]
        roi_blurred = cv2.GaussianBlur(roi, (11, 11), sigmaX=0)
        roi_resized = resize_and_pad(roi_blurred)

       # roi_resized = cv2.bitwise_not(roi_resized)
       # roi_resized = zoom_image(roi_resized)
        symbol = Image.fromarray(roi_resized)
        #symbol=symbol.filter(ImageFilter.GaussianBlur(5))

        if True:
            #os.makedirs("segmented_debug", exist_ok=True)
            symbol.save(f"segmented_debug/symbol_{i}.png")

        symbol_images.append(symbol)

    return symbol_images

