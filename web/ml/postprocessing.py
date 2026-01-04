from PIL import Image
import os
import random

def number_to_image(number, digits_path="data/digits_clean"):
    digits = str(number)
    images = []

    for digit in digits:
        variant = random.randint(0, 9)
        filename = f"{digit}_{variant}.png"
        path = os.path.join(digits_path, filename)

        if not os.path.exists(path):
            raise FileNotFoundError(path)

        images.append(Image.open(path).convert("L"))

    widths, heights = zip(*(img.size for img in images))
    final = Image.new("L", (sum(widths), max(heights)), 255)

    x = 0
    for img in images:
        final.paste(img, (x, 0))
        x += img.width

    return final
