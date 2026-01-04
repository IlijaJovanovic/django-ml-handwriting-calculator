from tensorflow.keras.models import load_model
import numpy as np

LABELS = {
    0:'0',1:'1',2:'2',3:'3',4:'4',
    5:'5',6:'6',7:'7',8:'8',9:'9',
    10:'+',11:'-',12:'*'
}

MODEL = load_model("ml_models/math_cnn.keras")

def model_predict(image):
    img = np.array(image).astype("float32") / 255.0
    img = img.reshape(1, 28, 28, 1)
    preds = MODEL.predict(img)
    return LABELS[int(np.argmax(preds))]

def evaluate_expression(symbols):
    expr = ''.join(symbols)
    try:
        return eval(expr)
    except Exception as e:
        return f"{e}"
