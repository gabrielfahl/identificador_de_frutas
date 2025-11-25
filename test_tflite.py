import numpy as np
import tensorflow as tf
import json
from PIL import Image
import os

# CONFIGURAÇÕES

MODEL_PATH = 'modelo_frutas.tflite'
CLASSES_PATH = 'classes.json'

IMAGE_PATH = r'D:\projetoiafrutas\testando\fruta.jpg' 

def load_tflite_model(model_path):
    # Carrega o interpretador (Simula o que vai rodar na nuvem)
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    return interpreter, input_details, output_details

def predict_image(interpreter, input_details, output_details, image_path):
    # 1. Carregar e Pré-processar a Imagem
    if not os.path.exists(image_path):
        return f"❌ Erro: Imagem não encontrada: {image_path}"

    img = Image.open(image_path).convert('RGB')
    
    # Pega o tamanho esperado pelo modelo automaticamente (ex: 100x100)
    input_shape = input_details[0]['shape']
    height = input_shape[1]
    width = input_shape[2]
    
    img = img.resize((width, height))
    
    # Converter para Array NumPy
    input_data = np.array(img)
    
    # Adicionar dimensão do lote (Batch Dimension)
    # De (100, 100, 3) vira (1, 100, 100, 3)
    input_data = np.expand_dims(input_data, axis=0)
    
    # Verificar o tipo de dados esperado (Float32 ou Int8)
    input_dtype = input_details[0]['dtype']
    if input_dtype == np.float32:
        input_data = input_data.astype(np.float32)
    
    # 2. Rodar a Inferência
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    
    # 3. Pegar o Resultado
    output_data = interpreter.get_tensor(output_details[0]['index'])
    
    # O resultado é um array de probabilidades (ex: [0.1, 0.05, 0.9, ...])
    # Pegamos o índice do maior valor
    predicted_index = int(np.argmax(output_data))
    confidence = np.max(output_data) # Opcional: aplicar softmax se os valores não somarem 1
    
    return predicted_index, confidence

# EXECUÇÃO PRINCIPAL

print(f"🔄 Carregando modelo: {MODEL_PATH}...")
interpreter, input_details, output_details = load_tflite_model(MODEL_PATH)

print(f"📂 Carregando classes: {CLASSES_PATH}...")
with open(CLASSES_PATH, 'r') as f:
    class_names = json.load(f)

print(f"🖼️ Testando imagem: {IMAGE_PATH}")
idx, conf = predict_image(interpreter, input_details, output_details, IMAGE_PATH)

if isinstance(idx, int):
    fruta_detectada = class_names[idx]
    print("\n" + "="*30)
    print(f"RESULTADO DA IA: {fruta_detectada.upper()}")
    print("="*30)
    print(f"Índice da Classe: {idx}")
    print(f"Score Bruto: {conf:.2f}")
else:
    print(idx)