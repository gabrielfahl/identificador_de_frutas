import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

# 1. CONFIGURAÇÕES E GPU

# Tenta detectar GPU, mas segue executando se for CPU
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"GPU Detectada: {gpus[0]}")
else:
    print("GPU não detectada. O treino será feito na CPU.")

BATCH_SIZE = 16
IMG_HEIGHT = 100
IMG_WIDTH = 100
EPOCHS = 5

base_dir = r"D:\projetoiafrutas\fruits-360\fruits-360_100x100\fruits-360"

data_dir_train = os.path.join(base_dir, 'Training')
data_dir_test = os.path.join(base_dir, 'Test')

# Verificação de segurança
if not os.path.exists(data_dir_train):
    print(f"*PASTA NÃO ENCONTRADA: {data_dir_train}")
    print("Verifique se o caminho da pasta está exatamente correto.")
    exit()
else:
    print(f"*Diretório encontrado: {data_dir_train}")

# 2. CARREGAMENTO DOS DADOS

print("\n--- Carregando Dataset de Treino ---")
train_ds = tf.keras.utils.image_dataset_from_directory(
  data_dir_train,
  validation_split=0.2,
  subset="training",
  seed=123,
  image_size=(IMG_HEIGHT, IMG_WIDTH),
  batch_size=BATCH_SIZE)

print("\n--- Carregando Dataset de Validação ---")
val_ds = tf.keras.utils.image_dataset_from_directory(
  data_dir_train,
  validation_split=0.2,
  subset="validation",
  seed=123,
  image_size=(IMG_HEIGHT, IMG_WIDTH),
  batch_size=BATCH_SIZE)

class_names = train_ds.class_names
print(f"\nFrutas encontradas ({len(class_names)}): {class_names[:5]}...")

AUTOTUNE = tf.data.AUTOTUNE

#  Lê do disco (stream) sem lotar a RAM.
train_ds = train_ds.shuffle(500).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)

# 3. CONSTRUÇÃO DO MODELO (CNN)
model = Sequential([
  layers.Rescaling(1./255, input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
  
  layers.Conv2D(16, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  
  layers.Conv2D(32, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  
  layers.Conv2D(64, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  
  layers.Dropout(0.2),
  
  layers.Flatten(),
  layers.Dense(128, activation='relu'),
  layers.Dense(len(class_names))
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

model.summary()

# 4. TREINAMENTO
print("\n--- Iniciando Treinamento ---")
history = model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=EPOCHS
)

# 5. GERAR GRÁFICOS
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(EPOCHS)

plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Acurácia de Treino')
plt.plot(epochs_range, val_acc, label='Acurácia de Validação')
plt.legend(loc='lower right')
plt.title('Performance: Acurácia')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Perda de Treino')
plt.plot(epochs_range, val_loss, label='Perda de Validação')
plt.legend(loc='upper right')
plt.title('Performance: Perda (Loss)')

plt.savefig('grafico_performance.png')
print("\n* Gráfico salvo como 'grafico_performance.png'")

# 6. EXPORTAÇÃO

# Salvar nomes
import json
with open('classes.json', 'w') as f:
    json.dump(class_names, f)
print("📄 Lista de frutas salva como 'classes.json'")

# Salvar modelo Keras
model.save('modelo_frutas.keras')
print("Modelo padrão salvo como 'modelo_frutas.keras'")

# Converter para TFLite
print("\n--- Convertendo para TFLite ---")
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

with open('modelo_frutas.tflite', 'wb') as f:
    f.write(tflite_model)
print("Modelo otimizado salvo como 'modelo_frutas.tflite'")
print("\n PROCESSO CONCLUÍDO COM SUCESSO!")