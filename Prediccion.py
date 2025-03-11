import cv2
import mediapipe as mp
import time
import json
import numpy as np
import tensorflow as tf
import joblib
import pandas as pd
import os
from sklearn.preprocessing import StandardScaler
from LecturaJson import GuardadCSv
from LecturaVideos import LecturaVideos


model_path = "Letras.h5"
rutaDataSet = "DataSet/Entrenamiento"
scaler_path = "scaler.pkl"
rutaCSV = "correcciones.csv"
clases = ["A","B","C","D","E","F","G","H","Z"]
frames_data = []
puntosDer = []
puntosIzq = []

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

gaurdar = GuardadCSv()

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


def clasificacion(json_input):
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


def calcular_pendiente(punto1, punto2):
    x1, y1 = punto1
    x2, y2 = punto2
    if x2 - x1 == 0:  # Evitar división por cero
        return float("inf")  # Pendiente infinita (línea vertical)
    return (y2 - y1) / (x2 - x1)

def desplazamiento(puntos1, puntos2):
    des = []
    for i in enumerate(puntos1):
        x = puntos2[i[0]][0] - puntos1[i[0]][0]
        y = puntos2[i[0]][1] - puntos1[i[0]][1]
        des.append([x, y])
    return des

def variacion(jason, puntosDer, puntosIzq, tempPuntosDer, tempPuntosIzq):
    jason ["datos_brazos"]["variable_Derecha"] = {
        "pulgar" : jason["datos_brazos"]["Mano Derecha"]["0_1"]-jason["datos_brazos"]["Mano Derecha"]["1_2"]-jason["datos_brazos"]["Mano Derecha"]["2_3"]-jason["datos_brazos"]["Mano Derecha"]["3_4"],
        "indice" : jason["datos_brazos"]["Mano Derecha"]["0_5"]-jason["datos_brazos"]["Mano Derecha"]["5_6"]-jason["datos_brazos"]["Mano Derecha"]["6_7"]-jason["datos_brazos"]["Mano Derecha"]["7_8"],
        "medio": jason["datos_brazos"]["Mano Derecha"]["9_10"]-jason["datos_brazos"]["Mano Derecha"]["10_11"]-jason["datos_brazos"]["Mano Derecha"]["11_12"],
        "anular":jason["datos_brazos"]["Mano Derecha"]["13_14"]-jason["datos_brazos"]["Mano Derecha"]["14_15"]-jason["datos_brazos"]["Mano Derecha"]["15_16"],
        "menique":jason["datos_brazos"]["Mano Derecha"]["0_17"]-jason["datos_brazos"]["Mano Derecha"]["17_18"]-jason["datos_brazos"]["Mano Derecha"]["18_19"]-jason["datos_brazos"]["Mano Derecha"]["19_20"],
        "0_X": desplazamiento(tempPuntosDer, puntosDer)[0][0],
        "0_Y": desplazamiento(tempPuntosDer, puntosDer)[0][1],
        "4_X": desplazamiento(tempPuntosDer, puntosDer)[1][0],
        "4_Y": desplazamiento(tempPuntosDer, puntosDer)[1][1],
        "8_X": desplazamiento(tempPuntosDer, puntosDer)[2][0],
        "8_Y": desplazamiento(tempPuntosDer, puntosDer)[2][1],
        "12_X": desplazamiento(tempPuntosDer, puntosDer)[3][0],
        "12_Y": desplazamiento(tempPuntosDer, puntosDer)[3][1],
        "16_X": desplazamiento(tempPuntosDer, puntosDer)[4][0],
        "16_Y": desplazamiento(tempPuntosDer, puntosDer)[4][1],
        "20_X": desplazamiento(tempPuntosDer, puntosDer)[5][0],
        "20_Y": desplazamiento(tempPuntosDer, puntosDer)[5][1]
    
    }

    jason ["datos_brazos"]["variable_Izquierda"] = {
        "pulgar" : jason["datos_brazos"]["Mano Izquierda"]["0_1"]-jason["datos_brazos"]["Mano Izquierda"]["1_2"]-jason["datos_brazos"]["Mano Izquierda"]["2_3"]-jason["datos_brazos"]["Mano Izquierda"]["3_4"],
        "indice" : jason["datos_brazos"]["Mano Izquierda"]["0_5"]-jason["datos_brazos"]["Mano Izquierda"]["5_6"]-jason["datos_brazos"]["Mano Izquierda"]["6_7"]-jason["datos_brazos"]["Mano Izquierda"]["7_8"],
        "medio": jason["datos_brazos"]["Mano Izquierda"]["9_10"]-jason["datos_brazos"]["Mano Izquierda"]["10_11"]-jason["datos_brazos"]["Mano Izquierda"]["11_12"],
        "anular":jason["datos_brazos"]["Mano Izquierda"]["13_14"]-jason["datos_brazos"]["Mano Izquierda"]["14_15"]-jason["datos_brazos"]["Mano Izquierda"]["15_16"],
        "menique":jason["datos_brazos"]["Mano Izquierda"]["0_17"]-jason["datos_brazos"]["Mano Izquierda"]["17_18"]-jason["datos_brazos"]["Mano Izquierda"]["18_19"]-jason["datos_brazos"]["Mano Izquierda"]["19_20"],
        "0_X": desplazamiento(tempPuntosIzq, puntosIzq)[0][0],
        "0_Y": desplazamiento(tempPuntosIzq, puntosIzq)[0][1],
        "4_X": desplazamiento(tempPuntosIzq, puntosIzq)[1][0],
        "4_Y": desplazamiento(tempPuntosIzq, puntosIzq)[1][1],
        "8_X": desplazamiento(tempPuntosIzq, puntosIzq)[2][0],
        "8_Y": desplazamiento(tempPuntosIzq, puntosIzq)[2][1],
        "12_X": desplazamiento(tempPuntosIzq, puntosIzq)[3][0],
        "12_Y": desplazamiento(tempPuntosIzq, puntosIzq)[3][1],
        "16_X": desplazamiento(tempPuntosIzq, puntosIzq)[4][0],
        "16_Y": desplazamiento(tempPuntosIzq, puntosIzq)[4][1],
        "20_X": desplazamiento(tempPuntosIzq, puntosIzq)[5][0],
        "20_Y": desplazamiento(tempPuntosIzq, puntosIzq)[5][1]

    }


    return jason

def mostrarCamara(prediccion, confianza, frame):
    cv2.putText(frame,f"{prediccion} ({confianza:.0%})",(10, 50),cv2.FONT_HERSHEY_SIMPLEX,1,(0, 255, 0),2,)
    cv2.imshow("Prediccion", frame)

def camara(opc, last_capture_time, cap, nomClase, puntosDer, puntosIzq, tempPuntosDer, tempPuntosIzq):
     with mp_holistic.Holistic(static_image_mode=False, model_complexity=1) as holistic:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

        # Calcular el tiempo transcurrido desde el último fotograma capturado
            current_time = time.time()
            elapsed_time = current_time - last_capture_time

        # Capturar solo 5 fotogramas por segundo
            if elapsed_time >= 0.2:  # 1 segundo / 5 = 0.2 segundos por fotograma
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = holistic.process(frame_rgb)

            # Diccionario para almacenar los datos del fotograma actual
                frame_data = {"id": len(frames_data) + 1, "datos_brazos": {}}

            # Postura (brazos)
                if results.pose_landmarks:
                    frame_data["datos_brazos"]["Brazo Derecho"] = {
                    "Brazo": calcular_pendiente((results.pose_landmarks.landmark[11].x,results.pose_landmarks.landmark[11].y,),(results.pose_landmarks.landmark[13].x,results.pose_landmarks.landmark[13].y,),),
                    "Antebrazo": calcular_pendiente((results.pose_landmarks.landmark[13].x,results.pose_landmarks.landmark[13].y,),(results.pose_landmarks.landmark[15].x,results.pose_landmarks.landmark[15].y,),),
                }
                    frame_data["datos_brazos"]["Brazo Izquierdo"] = {
                    "Brazo": calcular_pendiente((results.pose_landmarks.landmark[12].x,results.pose_landmarks.landmark[12].y,),(results.pose_landmarks.landmark[14].x,results.pose_landmarks.landmark[14].y,),),
                    "Antebrazo": calcular_pendiente((results.pose_landmarks.landmark[14].x,results.pose_landmarks.landmark[14].y,),(results.pose_landmarks.landmark[16].x,results.pose_landmarks.landmark[16].y,),),
                }
                else:
                    frame_data["datos_brazos"]["Brazo Derecho"] = {
                    "Brazo": 0,
                    "Antebrazo": 0,
                }
                    frame_data["datos_brazos"]["Brazo Izquierdo"] = {
                    "Brazo": 0,
                    "Antebrazo": 0,
                }

            # Mano derecha
                if results.right_hand_landmarks:
                    #===========================DESPLAZAMIENTO===========================================================
                    puntosDer.append([results.right_hand_landmarks.landmark[0].x, results.right_hand_landmarks.landmark[0].y])
                    puntosDer.append([results.right_hand_landmarks.landmark[4].x, results.right_hand_landmarks.landmark[4].y])
                    puntosDer.append([results.right_hand_landmarks.landmark[8].x, results.right_hand_landmarks.landmark[8].y])
                    puntosDer.append([results.right_hand_landmarks.landmark[12].x, results.right_hand_landmarks.landmark[12].y])
                    puntosDer.append([results.right_hand_landmarks.landmark[16].x, results.right_hand_landmarks.landmark[16].y])
                    puntosDer.append([results.right_hand_landmarks.landmark[20].x, results.right_hand_landmarks.landmark[20].y])
                    frame_data["datos_brazos"]["Mano Derecha"] = {
                    "0_1": calcular_pendiente((results.right_hand_landmarks.landmark[0].x,results.right_hand_landmarks.landmark[0].y,),(results.right_hand_landmarks.landmark[1].x, results.right_hand_landmarks.landmark[1].y,),),
                    "1_2": calcular_pendiente((results.right_hand_landmarks.landmark[1].x,results.right_hand_landmarks.landmark[1].y,),(results.right_hand_landmarks.landmark[2].x,results.right_hand_landmarks.landmark[2].y,),),
                    "2_3": calcular_pendiente((results.right_hand_landmarks.landmark[2].x, results.right_hand_landmarks.landmark[2].y,), ( results.right_hand_landmarks.landmark[3].x, results.right_hand_landmarks.landmark[3].y, ),),
                    "3_4": calcular_pendiente((results.right_hand_landmarks.landmark[3].x, results.right_hand_landmarks.landmark[3].y,),(results.right_hand_landmarks.landmark[4].x,results.right_hand_landmarks.landmark[4].y,),),
                    "0_5": calcular_pendiente( (results.right_hand_landmarks.landmark[0].x,results.right_hand_landmarks.landmark[0].y,),(results.right_hand_landmarks.landmark[5].x, results.right_hand_landmarks.landmark[5].y, ),),
                    "5_6": calcular_pendiente((results.right_hand_landmarks.landmark[5].x,results.right_hand_landmarks.landmark[5].y,),(results.right_hand_landmarks.landmark[6].x,results.right_hand_landmarks.landmark[6].y,),),
                    "6_7": calcular_pendiente((results.right_hand_landmarks.landmark[6].x,results.right_hand_landmarks.landmark[6].y, ),(results.right_hand_landmarks.landmark[7].x, results.right_hand_landmarks.landmark[7].y,),),
                    "7_8": calcular_pendiente((results.right_hand_landmarks.landmark[7].x,results.right_hand_landmarks.landmark[7].y,), (results.right_hand_landmarks.landmark[8].x, results.right_hand_landmarks.landmark[8].y,),),
                    "5_9": calcular_pendiente((results.right_hand_landmarks.landmark[5].x,results.right_hand_landmarks.landmark[5].y,),(results.right_hand_landmarks.landmark[9].x,results.right_hand_landmarks.landmark[9].y,),),
                    "9_10": calcular_pendiente((results.right_hand_landmarks.landmark[9].x,results.right_hand_landmarks.landmark[9].y,),(results.right_hand_landmarks.landmark[10].x,results.right_hand_landmarks.landmark[10].y,),),
                    "10_11": calcular_pendiente((results.right_hand_landmarks.landmark[10].x,results.right_hand_landmarks.landmark[10].y,), (results.right_hand_landmarks.landmark[11].x,results.right_hand_landmarks.landmark[11].y,), ),
                    "11_12": calcular_pendiente((results.right_hand_landmarks.landmark[11].x,results.right_hand_landmarks.landmark[11].y,),(results.right_hand_landmarks.landmark[12].x,results.right_hand_landmarks.landmark[12].y,),),
                    "9_13": calcular_pendiente((results.right_hand_landmarks.landmark[9].x,results.right_hand_landmarks.landmark[9].y,),(results.right_hand_landmarks.landmark[13].x,results.right_hand_landmarks.landmark[13].y,),),
                    "13_14": calcular_pendiente((results.right_hand_landmarks.landmark[13].x,results.right_hand_landmarks.landmark[13].y,),(results.right_hand_landmarks.landmark[14].x,results.right_hand_landmarks.landmark[14].y,),),
                    "14_15": calcular_pendiente((results.right_hand_landmarks.landmark[14].x,results.right_hand_landmarks.landmark[14].y,),(results.right_hand_landmarks.landmark[15].x,results.right_hand_landmarks.landmark[15].y,),),
                    "15_16": calcular_pendiente((results.right_hand_landmarks.landmark[15].x,results.right_hand_landmarks.landmark[15].y,),(results.right_hand_landmarks.landmark[16].x,results.right_hand_landmarks.landmark[16].y,),),
                    "13_17": calcular_pendiente((results.right_hand_landmarks.landmark[13].x,results.right_hand_landmarks.landmark[13].y,),(results.right_hand_landmarks.landmark[17].x,results.right_hand_landmarks.landmark[17].y,),),
                    "0_17": calcular_pendiente((results.right_hand_landmarks.landmark[0].x,results.right_hand_landmarks.landmark[0].y,),(results.right_hand_landmarks.landmark[17].x,results.right_hand_landmarks.landmark[17].y,),),
                    "17_18": calcular_pendiente((results.right_hand_landmarks.landmark[17].x,results.right_hand_landmarks.landmark[17].y,),(results.right_hand_landmarks.landmark[18].x,results.right_hand_landmarks.landmark[18].y,),),
                    "18_19": calcular_pendiente((results.right_hand_landmarks.landmark[18].x,results.right_hand_landmarks.landmark[18].y,),(results.right_hand_landmarks.landmark[19].x, results.right_hand_landmarks.landmark[19].y,),),
                    "19_20": calcular_pendiente((results.right_hand_landmarks.landmark[19].x,results.right_hand_landmarks.landmark[19].y,),(results.right_hand_landmarks.landmark[20].x,results.right_hand_landmarks.landmark[20].y,),),
                }

                else:
                    puntosDer = [
                        [0,0],
                        [0,0],
                        [0,0],
                        [0,0],
                        [0,0],
                        [0,0]
                        
                    ]
                    frame_data["datos_brazos"]["Mano Derecha"] = {
                    "0_1": 0,
                    "1_2": 0,
                    "2_3": 0,
                    "3_4": 0,
                    "0_5": 0,
                    "5_6": 0,
                    "6_7": 0,
                    "7_8": 0,
                    "5_9": 0,
                    "9_10": 0,
                    "10_11": 0,
                    "11_12": 0,
                    "9_13": 0,
                    "13_14": 0,
                    "14_15": 0,
                    "15_16": 0,
                    "13_17": 0,
                    "0_17": 0,
                    "17_18": 0,
                    "18_19": 0,
                    "19_20": 0,
                }

            # Mano izquierda
                if results.left_hand_landmarks:
                    #==============================DESPLAZAMIENTO=================================
                    puntosIzq.append([results.left_hand_landmarks.landmark[0].x,results.left_hand_landmarks.landmark[0].y])
                    puntosIzq.append([results.left_hand_landmarks.landmark[4].x,results.left_hand_landmarks.landmark[4].y])
                    puntosIzq.append([results.left_hand_landmarks.landmark[8].x,results.left_hand_landmarks.landmark[8].y])
                    puntosIzq.append([results.left_hand_landmarks.landmark[12].x,results.left_hand_landmarks.landmark[12].y])
                    puntosIzq.append([results.left_hand_landmarks.landmark[16].x,results.left_hand_landmarks.landmark[16].y])
                    puntosIzq.append([results.left_hand_landmarks.landmark[20].x,results.left_hand_landmarks.landmark[20].y])

                    frame_data["datos_brazos"]["Mano Izquierda"] = {
                    "0_1": calcular_pendiente((results.left_hand_landmarks.landmark[0].x,results.left_hand_landmarks.landmark[0].y,),(results.left_hand_landmarks.landmark[1].x,results.left_hand_landmarks.landmark[1].y,),),
                    "1_2": calcular_pendiente((results.left_hand_landmarks.landmark[1].x,results.left_hand_landmarks.landmark[1].y,),(results.left_hand_landmarks.landmark[2].x,results.left_hand_landmarks.landmark[2].y,),),
                    "2_3": calcular_pendiente((results.left_hand_landmarks.landmark[2].x,results.left_hand_landmarks.landmark[2].y,),(results.left_hand_landmarks.landmark[3].x,results.left_hand_landmarks.landmark[3].y,),),
                    "3_4": calcular_pendiente((results.left_hand_landmarks.landmark[3].x,results.left_hand_landmarks.landmark[3].y,),(results.left_hand_landmarks.landmark[4].x,results.left_hand_landmarks.landmark[4].y,),),
                    "0_5": calcular_pendiente((results.left_hand_landmarks.landmark[0].x,results.left_hand_landmarks.landmark[0].y,),(results.left_hand_landmarks.landmark[5].x,results.left_hand_landmarks.landmark[5].y,),),
                    "5_6": calcular_pendiente((results.left_hand_landmarks.landmark[5].x,results.left_hand_landmarks.landmark[5].y,),(results.left_hand_landmarks.landmark[6].x,results.left_hand_landmarks.landmark[6].y,),),
                    "6_7": calcular_pendiente((results.left_hand_landmarks.landmark[6].x,results.left_hand_landmarks.landmark[6].y,),(results.left_hand_landmarks.landmark[7].x,results.left_hand_landmarks.landmark[7].y,),),
                    "7_8": calcular_pendiente((results.left_hand_landmarks.landmark[7].x,results.left_hand_landmarks.landmark[7].y,),(results.left_hand_landmarks.landmark[8].x,results.left_hand_landmarks.landmark[8].y,),),
                    "5_9": calcular_pendiente((results.left_hand_landmarks.landmark[5].x,results.left_hand_landmarks.landmark[5].y,),(results.left_hand_landmarks.landmark[9].x,results.left_hand_landmarks.landmark[9].y,),),
                    "9_10": calcular_pendiente((results.left_hand_landmarks.landmark[9].x,results.left_hand_landmarks.landmark[9].y,),(results.left_hand_landmarks.landmark[10].x,results.left_hand_landmarks.landmark[10].y,),),
                    "10_11": calcular_pendiente((results.left_hand_landmarks.landmark[10].x,results.left_hand_landmarks.landmark[10].y,),(results.left_hand_landmarks.landmark[11].x,results.left_hand_landmarks.landmark[11].y,),),
                    "11_12": calcular_pendiente((results.left_hand_landmarks.landmark[11].x,results.left_hand_landmarks.landmark[11].y,),(results.left_hand_landmarks.landmark[12].x,results.left_hand_landmarks.landmark[12].y,),),
                    "9_13": calcular_pendiente((results.left_hand_landmarks.landmark[9].x,results.left_hand_landmarks.landmark[9].y,),(results.left_hand_landmarks.landmark[13].x,results.left_hand_landmarks.landmark[13].y,),),
                    "13_14": calcular_pendiente((results.left_hand_landmarks.landmark[13].x,results.left_hand_landmarks.landmark[13].y,),(results.left_hand_landmarks.landmark[14].x,results.left_hand_landmarks.landmark[14].y,),),
                    "14_15": calcular_pendiente((results.left_hand_landmarks.landmark[14].x,results.left_hand_landmarks.landmark[14].y,),(results.left_hand_landmarks.landmark[15].x,results.left_hand_landmarks.landmark[15].y,),),
                    "15_16": calcular_pendiente((results.left_hand_landmarks.landmark[15].x,results.left_hand_landmarks.landmark[15].y,),(results.left_hand_landmarks.landmark[16].x,results.left_hand_landmarks.landmark[16].y,),),
                    "13_17": calcular_pendiente((results.left_hand_landmarks.landmark[13].x,results.left_hand_landmarks.landmark[13].y,),(results.left_hand_landmarks.landmark[17].x,results.left_hand_landmarks.landmark[17].y,),),
                    "0_17": calcular_pendiente((results.left_hand_landmarks.landmark[0].x,results.left_hand_landmarks.landmark[0].y,),(results.left_hand_landmarks.landmark[17].x,results.left_hand_landmarks.landmark[17].y,),),
                    "17_18": calcular_pendiente((results.left_hand_landmarks.landmark[17].x,results.left_hand_landmarks.landmark[17].y,),(results.left_hand_landmarks.landmark[18].x,results.left_hand_landmarks.landmark[18].y,),),
                    "18_19": calcular_pendiente((results.left_hand_landmarks.landmark[18].x,results.left_hand_landmarks.landmark[18].y,),(results.left_hand_landmarks.landmark[19].x,results.left_hand_landmarks.landmark[19].y,),),
                    "19_20": calcular_pendiente((results.left_hand_landmarks.landmark[19].x,results.left_hand_landmarks.landmark[19].y,),(results.left_hand_landmarks.landmark[20].x,results.left_hand_landmarks.landmark[20].y,),),
                }
                else:
                    puntosIzq = [
                        [0,0],
                        [0,0],
                        [0,0],
                        [0,0],
                        [0,0],
                        [0,0]
                        
                    ]
                    frame_data["datos_brazos"]["Mano Izquierda"] = {
                    "0_1": 0,
                    "1_2": 0,
                    "2_3": 0,
                    "3_4": 0,
                    "0_5": 0,
                    "5_6": 0,
                    "6_7": 0,
                    "7_8": 0,
                    "5_9": 0,
                    "9_10": 0,
                    "10_11": 0,
                    "11_12": 0,
                    "9_13": 0,
                    "13_14": 0,
                    "14_15": 0,
                    "15_16": 0,
                    "13_17": 0,
                    "0_17": 0,
                    "17_18": 0,
                    "18_19": 0,
                    "19_20": 0,
                }

            # Voltear el frame horizontalmente
                frame = cv2.flip(frame, 1)

            # Actualizar el tiempo del último fotograma capturado
                last_capture_time = current_time

                #json_formateado = json.dumps(frame_data, indent=4, ensure_ascii=False)

                #print(json_formateado)
                #json_formateado = json.dumps(variacion(frame_data), indent=4, ensure_ascii=False)
                #print(variacion(frame_data)["datos_brazos"]["Mano Izquierda"]["variable"]
                print("====================================================")
                print("====================================================")
                print(puntosDer)
                print(len(puntosDer))
                print(len(puntosIzq))
                print(len(tempPuntosDer))
                print(len(tempPuntosIzq))
                print("====================================================")
                
                frame_data = variacion(frame_data, puntosDer, puntosIzq, tempPuntosDer, tempPuntosIzq)

                tempPuntosDer = puntosDer
                tempPuntosIzq = puntosIzq
                puntosDer = []
                puntosIzq = []
                print(len(puntosDer))
                print(len(puntosIzq))
                print(len(tempPuntosDer))
                print(len(tempPuntosIzq))
                print("====================================================")
                print("====================================================")

            # ====================================================================
                prediccion, confianza = clasificacion(frame_data)
                
                print(f"Predicción: {prediccion} (Confianza: {confianza:.2f})")
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
    
    leerVideo = LecturaVideos()
    for ruta in leerVideo.extraccion(rutaDataSet):
        camara(2, last_capture_time, cv2.VideoCapture(ruta[0]), ruta[1], puntosDer, puntosIzq, tempPuntosDer, tempPuntosIzq)
        #print(ruta[1])
    #gaurdar.Entrenar(rutaCSV, model_path, clases)
    #Este de aqui es para entrenar lo que se tiene en el csv, descomentar solo para reeentrenar
    #if ( os.path.exists(rutaCSV)and input("¿Reentrenar el modelo? (s/n): ").strip().lower() == "s"): gaurdar.reentrenar_modelo(rutaCSV, model_path, model, scaler, clases)
    


    #captura = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    #camara(1, last_capture_time, captura, "")

