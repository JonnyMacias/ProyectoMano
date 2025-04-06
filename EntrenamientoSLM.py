import csv
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import joblib
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.callbacks import EarlyStopping

# Cargar los datos desde CSV
X = []
y = []

with open('corpues.csv', newline='', encoding='utf-8') as csvfile:
    reader = csv.reader(csvfile)
    next(reader)

    for row in reader:
        X.append(row[:-1]) 
        y.append(row[-1]) 

X = np.array(X, dtype=np.float32)

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

y = to_categorical(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

joblib.dump(scaler, 'scaler.pkl')

X_train_scaled = X_train_scaled.reshape((X_train_scaled.shape[0], 1, X_train_scaled.shape[1]))
X_test_scaled = X_test_scaled.reshape((X_test_scaled.shape[0], 1, X_test_scaled.shape[1]))

modelo = Sequential([
    LSTM(128, activation='tanh', return_sequences=True, input_shape=(1, X.shape[1])),
    BatchNormalization(),  
    Dropout(0.2),

    LSTM(64, activation='tanh', return_sequences=True),
    BatchNormalization(),
    Dropout(0.2),

    LSTM(32, activation='tanh'),
    Dropout(0.2),

    Dense(64, activation='relu'),  
    Dropout(0.2),

    Dense(y.shape[1], activation='softmax') 
])

modelo.compile(optimizer=RMSprop(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

modelo.fit(X_train_scaled, y_train, epochs=150, batch_size=16, validation_data=(X_test_scaled, y_test))
#modelo.fit(X_train_scaled, y_train, epochs=150, batch_size=16, validation_data=(X_test_scaled, y_test), callbacks=[early_stopping])

# Guardar el modelo entrenado
modelo.save('Letras_LSTM.h5')
print("Modelo LSTM con más capas ocultas guardado.")
