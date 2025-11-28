import tensorflow as tf
import numpy as np
from PIL import Image
import json
from io import BytesIO

# Carrega o modelo tflite
interpreter = tf.lite.Interpreter(model_path="modelo_frutas.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Carrega nomes das frutas
with open("classes.json", "r") as f:
    class_names = json.load(f)

def predict_from_bytes(image_bytes: bytes):
    img = Image.open(BytesIO(image_bytes)).convert("RGB")

    height = input_details[0]["shape"][1]
    width = input_details[0]["shape"][2]

    img = img.resize((width, height))

    input_data = np.expand_dims(np.array(img), axis=0).astype(np.float32)

    interpreter.set_tensor(input_details[0]["index"], input_data)
    interpreter.invoke()

    output_data = interpreter.get_tensor(output_details[0]["index"])

    idx = int(np.argmax(output_data))
    score = float(np.max(output_data))

    return {
        "indice": idx,
        "fruta": class_names[idx],
        "score": score
    }