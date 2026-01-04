import io
import base64

from .preprocessing import segment_characters
from .inference import model_predict, evaluate_expression
from .postprocessing import number_to_image


def pil_to_base64(img):
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def predict_expression(image):
    # 1. Segmentacija
    chars = segment_characters(image)

    if not chars:
        return {"error": "No symbols detected"}

    # 2. Segmentirane slike -> base64
    segmented_images_b64 = [pil_to_base64(img) for img in chars]

    # 3. Predikcija simbola
    symbols = [model_predict(img) for img in chars]

    # 4. Formiranje i evaluacija izraza
    expression = "".join(symbols)
    result = evaluate_expression(symbols)

    # 5. Generisanje slike rezultata
    result_img = number_to_image(result)
    result_img_b64 = pil_to_base64(result_img)

    return {
        "segmented_images": segmented_images_b64,
        "symbols": symbols,
        "expression": expression,
        "result": result,
        "result_image_base64": result_img_b64
    }
