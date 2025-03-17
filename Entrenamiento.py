import csv
import numpy as np
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import StandardScaler
import joblib


X = []
y = []

with open('correcciones.csv', newline='', encoding='utf-8') as csvfile:
    reader = csv.reader(csvfile)
    next(reader)  # Omitir la cabecera si tiene

    for row in reader:
        X.append(row[:-1])  # Todas las columnas excepto la última (features)
        y.append(row[-1])   # Última columna (clase)

# Convertir X en array de NumPy
X = np.array(X, dtype=np.float32)

# Convertir y en etiquetas numéricas
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Convertir y a one-hot encoding (si es clasificación multiclase)
y = to_categorical(y)

# Ver las primeras 10 etiquetas convertidas

# División en conjunto de entrenamiento y prueba
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
joblib.dump(scaler, 'scaler.pkl')
# Definir el modelo
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

modelo = Sequential([
    Dense(64, activation='relu', input_shape=(X.shape[1],)),
    Dense(32, activation='relu'),
    Dense(y.shape[1], activation='softmax')
])

# Compilar el modelo
modelo.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Entrenar el modelo
modelo.fit(X_train, y_train, epochs=50, batch_size=8, validation_data=(X_test, y_test))


modelo.save('Letras.h5')
print("Modelo reentrenado y guardado.")
