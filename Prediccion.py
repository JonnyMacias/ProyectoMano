import cv2
import mediapipe as mp
import time
import numpy as np
import tensorflow as tf
import joblib
import pandas as pd
import os
from sklearn.preprocessing import StandardScaler
from LecturaJson import GuardadCSv
from LecturaDataSet import LecturaDataSet
from PreProcesamiento import PreProcesamiento


model_path = "Letras_LSTM.h5"
rutaDataSet = "DataSet/Entrenamiento"
scaler_path = "scaler.pkl"
rutaCSV = "correcciones.csv"
clases = ["A","B","C","D","E","F","G","H","I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y"]
frames_data = []

gaurdar = GuardadCSv()
preProceso = PreProcesamiento()

model = tf.keras.models.load_model(model_path)
scaler = joblib.load(scaler_path)

mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic
mp_face_mesh = mp.solutions.face_mesh
mp_pose = mp.solutions.pose
FACE_CONNECTIONS = mp_face_mesh.FACEMESH_TESSELATION

start_time = time.time()
frame_counter = 0
last_capture_time = start_time

def getpuntos():
    puntosDer = [
        [0,0],
        [0,0],
        [0,0],
        [0,0],
        [0,0],
        [0,0]
    ]
    puntosIzq = [
        [0,0],
        [0,0],
        [0,0],
        [0,0],
        [0,0],
        [0,0]
    ]
    tempPuntosDer = [
        [0,0],
        [0,0],
        [0,0],
        [0,0],
        [0,0],
        [0,0]
        
    ]
    tempPuntosIzq = [
        [0,0],
        [0,0],
        [0,0],
        [0,0],
        [0,0],
        [0,0]
    ]
    return puntosDer, puntosIzq, tempPuntosDer, tempPuntosIzq


def clasificacion(json_input):
    try:
        x = gaurdar.extraccion(json_input)  # Extraer características desde el JSON
        x = np.array(x)

        # 🔹 Asegurar que la forma es compatible con LSTM
        x = x.reshape(1, -1)  # Convertir a 2D (1, features)
        x = scaler.transform(x)  # Escalar los datos
        x = x.reshape(1, 1, -1)  # 🔥 Convertir a 3D para LSTM (1, timesteps=1, features)

        print("==============================================================================")
        print("Forma de X antes de la predicción:" + str(x.shape))  # Verificar que sea (1, 1, features)

        # Hacer la predicción
        predicciones = model.predict(x, verbose=0)
        clase_idx = np.argmax(predicciones)
        confianza = np.max(predicciones)

        # Validar confianza
        if confianza < 0.5:
            return "Clase no determinada", confianza
        if clase_idx >= len(clases):
            return "Error: Clase desconocida", confianza

        return clases[clase_idx], confianza

    except Exception as e:
        print(f"Error en clasificación: {str(e)}")
        return "Error en procesamiento", 0.0

def clasificacionM(json_input):
    try:
        x = gaurdar.extraccion(json_input)
        x = np.array(x)
        x = x.reshape(1, -1)
        print("==============================================================================")
        x = scaler.transform(x)
        print("Forma de X antes de la predicción:" + str(x.shape))
        predicciones = model.predict(x, verbose=0)
        clase_idx = np.argmax(predicciones)
        confianza = np.max(predicciones)

        if confianza < 0.5:
            return "Clase no determinada", confianza
        if clase_idx >= len(clases):
            return "Error: Clase desconocida", confianza

        return clases[clase_idx], confianza

    except Exception as e:
        print(f"Error en clasificación: {str(e)}")
        return "Error en procesamiento", 0.0

def mostrarCamara(prediccion, confianza, frame):
    cv2.putText(frame,f"{prediccion} ({confianza:.0%})",(10, 50),cv2.FONT_HERSHEY_SIMPLEX,1,(0, 255, 0),2,)
    cv2.imshow("Prediccion", frame)


def imagen(ruta, puntosDer, puntosIzq, nomClase):
    with mp_holistic.Holistic(
        static_image_mode=True,
        model_complexity=2) as holistic:

        image = cv2.imread(ruta)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        results = holistic.process(image_rgb)
        frame_data = {"id": len(frames_data) + 1, "datos_brazos": {}}
        frame_data, puntosDer, puntosIzq = preProceso.pendiente(results, frame_data, puntosDer, puntosIzq)
        puntosDer, puntosIzq, tempPuntosDer, tempPuntosIzq = getpuntos()
        frame_data = preProceso.variacion(frame_data, puntosDer, puntosIzq, tempPuntosDer, tempPuntosIzq)
        gaurdar.guardadDatos(frame_data, nomClase, rutaCSV)

        cv2.waitKey(0)
    cv2.destroyAllWindows()

def camara(opc, last_capture_time, cap, nomClase, puntosDer, puntosIzq, tempPuntosDer, tempPuntosIzq):
     with mp_holistic.Holistic(static_image_mode=False, model_complexity=1) as holistic:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            current_time = time.time()
            elapsed_time = current_time - last_capture_time
            if elapsed_time >= 0.2:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = holistic.process(frame_rgb)


                tempPuntosDer = puntosDer
                tempPuntosIzq = puntosIzq
                puntosDer = []
                puntosIzq = []
                frame_data = {"id": len(frames_data) + 1, "datos_brazos": {}}
                frame = cv2.flip(frame, 1)
                last_capture_time = current_time
                frame_data, puntosDer, puntosIzq = preProceso.pendiente(results, frame_data, puntosDer, puntosIzq)
                #========================PRE-PROCESAMIENTO=============================
                print(preProceso.movimiento(frame_data))
                preProceso.setJson_P(frame_data)
                #======================================================================
                frame_data = preProceso.variacion(frame_data, puntosDer, puntosIzq, tempPuntosDer, tempPuntosIzq)
                prediccion, confianza = clasificacion(frame_data)
                #========================IMPRESIONES EN CONSOLA=========================
                """json_formateado = json.dumps(frame_data, indent=4, ensure_ascii=False)
                print(json_formateado)"""
                #print(f"Predicción: {prediccion} (Confianza: {confianza:.2f})")
                #========================================================================
                if opc == 1:
                    mostrarCamara(prediccion, confianza, frame)
                elif opc == 2:
                    gaurdar.guardadDatos(frame_data, nomClase, rutaCSV)
                elif opc == 3:
                    key = gaurdar.validacion(prediccion, frame_data, rutaCSV, clases)
                    if  key == 27:
                       if ( os.path.exists(rutaCSV)and input("¿Reentrenar el modelo? (s/n): ").strip().lower() == "s"): gaurdar.reentrenar_modelo(rutaCSV, model_path, model, scaler, clases)
                       break
            if cv2.waitKey(1) & 0xFF == 27:
                break
        
     cap.release()
     cv2.destroyAllWindows()

if __name__ == '__main__':
    
    leerVideo = LecturaDataSet()
    lecturaImg = LecturaDataSet()
    """for ruta in leerVideo.extraccionVideo(rutaDataSet):
        preProceso.ReiniciarJson()
        puntosDer, puntosIzq, tempPuntosDer, tempPuntosIzq = getpuntos()
        camara(2, last_capture_time, cv2.VideoCapture(ruta[0]), ruta[1], puntosDer, puntosIzq, tempPuntosDer, tempPuntosIzq)
    """
    """for ruta in leerVideo.extraccionImagenes(rutaDataSet):
        preProceso.ReiniciarJson()
        puntosDer, puntosIzq, tempPuntosDer, tempPuntosIzq = getpuntos()
        print(ruta[1])
        imagen(ruta[0], puntosDer, puntosIzq, ruta[1])    
        """
        #print(ruta[1])
    #gaurdar.Entrenar(rutaCSV, model_path, clases)
    #Este de aqui es para entrenar lo que se tiene en el csv, descomentar solo para reeentrenar
    
    #if ( os.path.exists(rutaCSV)and input("¿Reentrenar el modelo? (s/n): ").strip().lower() == "s"): gaurdar.reentrenar_modelo(rutaCSV, model_path, model, scaler, clases)
    

    puntosDer, puntosIzq, tempPuntosDer, tempPuntosIzq = getpuntos()
    captura = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    camara(1, last_capture_time, captura, "", puntosDer, puntosIzq, tempPuntosDer, tempPuntosIzq)
