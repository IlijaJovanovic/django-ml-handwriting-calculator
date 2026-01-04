import numpy as np
from PIL import Image
import cv2
import os

def segment_characters(pil_img, debug=True):


    img = np.array(pil_img)

    _, thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    thresh=cv2.bitwise_not(thresh)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = sorted([cv2.boundingRect(c) for c in contours], key=lambda b: b[0])
    symbol_images = []

    for i, (x, y, w, h) in enumerate(boxes):
        if w < 10 or h < 10:
            continue

        roi = thresh[y:y+h, x:x+w]
        roi_blurred = cv2.GaussianBlur(roi, (11, 11), sigmaX=0)
        roi_resized = resize_and_pad(roi_blurred)

        symbol = Image.fromarray(roi_resized)

        if debug:
            os.makedirs("data/debug", exist_ok=True)
            symbol.save(f"data/debug/symbol_{i}.png")

        symbol_images.append(symbol)

    return symbol_images


def resize_and_pad(image, size=28, pad_value=0):
    h, w = image.shape
    scale = size / max(w, h)
    resized = cv2.resize(image, (int(w * scale), int(h * scale)))
    square = np.full((size, size), pad_value, dtype=np.uint8)
    y_offset = (size - resized.shape[0]) // 2
    x_offset = (size - resized.shape[1]) // 2
    square[y_offset:y_offset + resized.shape[0], x_offset:x_offset + resized.shape[1]] = resized
    return square


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