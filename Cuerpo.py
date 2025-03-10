import cv2
import mediapipe as mp
import time
import json
import math

# Función para calcular la pendiente entre dos puntos
def calcular_pendiente(punto1, punto2):
    x1, y1 = punto1
    x2, y2 = punto2
    if x2 - x1 == 0:  # Evitar división por cero
        return float('inf')  # Pendiente infinita (línea vertical)
    return (y2 - y1) / (x2 - x1)

mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic
mp_face_mesh = mp.solutions.face_mesh

# Face connections correctas
FACE_CONNECTIONS = mp_face_mesh.FACEMESH_TESSELATION  # O usa FACEMESH_CONTOURS si prefieres

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

mp_pose = mp.solutions.pose

# Lista para almacenar los datos de cada fotograma
frames_data = []

# Iniciar el temporizador
start_time = time.time()

# Contador para limitar la captura a 5 fotogramas por segundo
frame_counter = 0
last_capture_time = start_time

with mp_holistic.Holistic(
     static_image_mode=False,
     model_complexity=1) as holistic:

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
               frame_data = {
                   "id": len(frames_data) + 1,
                   "datos_brazos": {}
               }

               # Postura (brazos)
               if results.pose_landmarks:
                    frame_data["datos_brazos"]["Brazo Derecho"] = {
                         "Brazo": calcular_pendiente(
                             (results.pose_landmarks.landmark[11].x, results.pose_landmarks.landmark[11].y),
                             (results.pose_landmarks.landmark[13].x, results.pose_landmarks.landmark[13].y)
                         ),
                         "Antebrazo": calcular_pendiente(
                             (results.pose_landmarks.landmark[13].x, results.pose_landmarks.landmark[13].y),
                             (results.pose_landmarks.landmark[15].x, results.pose_landmarks.landmark[15].y)
                         )
                    }
                    frame_data["datos_brazos"]["Brazo Izquierdo"] = {
                         "Brazo": calcular_pendiente(
                             (results.pose_landmarks.landmark[12].x, results.pose_landmarks.landmark[12].y),
                             (results.pose_landmarks.landmark[14].x, results.pose_landmarks.landmark[14].y)
                         ),
                         "Antebrazo": calcular_pendiente(
                             (results.pose_landmarks.landmark[14].x, results.pose_landmarks.landmark[14].y),
                             (results.pose_landmarks.landmark[16].x, results.pose_landmarks.landmark[16].y)
                         )
                    }
               else:
                    frame_data["datos_brazos"]["Brazo Derecho"] = {
                         "Brazo": 0,
                         "Antebrazo": 0
                    }
                    frame_data["datos_brazos"]["Brazo Izquierdo"] = {
                         "Brazo": 0,
                         "Antebrazo": 0
                    }

               # Mano derecha
               if results.right_hand_landmarks:
                    frame_data["datos_brazos"]["Mano Derecha"] = {
                         "0_1": calcular_pendiente(
                             (results.right_hand_landmarks.landmark[0].x, results.right_hand_landmarks.landmark[0].y),
                             (results.right_hand_landmarks.landmark[1].x, results.right_hand_landmarks.landmark[1].y)
                         ),
                         "1_2": calcular_pendiente(
                             (results.right_hand_landmarks.landmark[1].x, results.right_hand_landmarks.landmark[1].y),
                             (results.right_hand_landmarks.landmark[2].x, results.right_hand_landmarks.landmark[2].y)
                         ),
                         "2_3": calcular_pendiente(
                              (results.right_hand_landmarks.landmark[2].x, results.right_hand_landmarks.landmark[2].y), 
                              (results.right_hand_landmarks.landmark[3].x, results.right_hand_landmarks.landmark[3].y)
                         ),
                         "3_4": calcular_pendiente(
                              (results.right_hand_landmarks.landmark[3].x, results.right_hand_landmarks.landmark[3].y),
                              (results.right_hand_landmarks.landmark[4].x, results.right_hand_landmarks.landmark[4].y)
                         ),
                         "0_5": calcular_pendiente(
                              (results.right_hand_landmarks.landmark[0].x, results.right_hand_landmarks.landmark[0].y),
                              (results.right_hand_landmarks.landmark[5].x, results.right_hand_landmarks.landmark[5].y)
                         ),
                         "5_6": calcular_pendiente(
                              (results.right_hand_landmarks.landmark[5].x, results.right_hand_landmarks.landmark[5].y),
                              (results.right_hand_landmarks.landmark[6].x, results.right_hand_landmarks.landmark[6].y)
                         ),
                         "6_7": calcular_pendiente(
                              (results.right_hand_landmarks.landmark[6].x, results.right_hand_landmarks.landmark[6].y),
                              (results.right_hand_landmarks.landmark[7].x, results.right_hand_landmarks.landmark[7].y)
                         ),
                         "7_8": calcular_pendiente(
                              (results.right_hand_landmarks.landmark[7].x, results.right_hand_landmarks.landmark[7].y),
                              (results.right_hand_landmarks.landmark[8].x, results.right_hand_landmarks.landmark[8].y)
                         ),
                         "5_9": calcular_pendiente(
                              (results.right_hand_landmarks.landmark[5].x, results.right_hand_landmarks.landmark[5].y),
                              (results.right_hand_landmarks.landmark[9].x, results.right_hand_landmarks.landmark[9].y)
                         ),
                         "9_10": calcular_pendiente(
                              (results.right_hand_landmarks.landmark[9].x, results.right_hand_landmarks.landmark[9].y),
                              (results.right_hand_landmarks.landmark[10].x, results.right_hand_landmarks.landmark[10].y)
                         ),
                         "10_11": calcular_pendiente(
                              (results.right_hand_landmarks.landmark[10].x, results.right_hand_landmarks.landmark[10].y),
                              (results.right_hand_landmarks.landmark[11].x, results.right_hand_landmarks.landmark[11].y)
                         ),
                         "11_12": calcular_pendiente(
                              (results.right_hand_landmarks.landmark[11].x, results.right_hand_landmarks.landmark[11].y),
                              (results.right_hand_landmarks.landmark[12].x, results.right_hand_landmarks.landmark[12].y)
                         ),
                         "9_13": calcular_pendiente(
                              (results.right_hand_landmarks.landmark[9].x, results.right_hand_landmarks.landmark[9].y),
                              (results.right_hand_landmarks.landmark[13].x, results.right_hand_landmarks.landmark[13].y)
                         ),
                         "13_14": calcular_pendiente(
                              (results.right_hand_landmarks.landmark[13].x, results.right_hand_landmarks.landmark[13].y),
                              (results.right_hand_landmarks.landmark[14].x, results.right_hand_landmarks.landmark[14].y)
                         ),
                         "14_15": calcular_pendiente(
                              (results.right_hand_landmarks.landmark[14].x, results.right_hand_landmarks.landmark[14].y),
                              (results.right_hand_landmarks.landmark[15].x, results.right_hand_landmarks.landmark[15].y)
                         ),
                         "15_16": calcular_pendiente(
                              (results.right_hand_landmarks.landmark[15].x, results.right_hand_landmarks.landmark[15].y),
                              (results.right_hand_landmarks.landmark[16].x, results.right_hand_landmarks.landmark[16].y)
                         ),
                         "13_17": calcular_pendiente(
                              (results.right_hand_landmarks.landmark[13].x, results.right_hand_landmarks.landmark[13].y),
                              (results.right_hand_landmarks.landmark[17].x, results.right_hand_landmarks.landmark[17].y)
                         ),
                         "0_17": calcular_pendiente(
                              (results.right_hand_landmarks.landmark[0].x, results.right_hand_landmarks.landmark[0].y),
                              (results.right_hand_landmarks.landmark[17].x, results.right_hand_landmarks.landmark[17].y)
                         ),
                         "17_18": calcular_pendiente(
                              (results.right_hand_landmarks.landmark[17].x, results.right_hand_landmarks.landmark[17].y),
                              (results.right_hand_landmarks.landmark[18].x, results.right_hand_landmarks.landmark[18].y)
                         ),
                         "18_19": calcular_pendiente(
                              (results.right_hand_landmarks.landmark[18].x, results.right_hand_landmarks.landmark[18].y),
                              (results.right_hand_landmarks.landmark[19].x, results.right_hand_landmarks.landmark[19].y)
                         ),
                         "19_20": calcular_pendiente(
                              (results.right_hand_landmarks.landmark[19].x, results.right_hand_landmarks.landmark[19].y),
                              (results.right_hand_landmarks.landmark[20].x, results.right_hand_landmarks.landmark[20].y)
                         )
                    }
                         
               else:
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
                    frame_data["datos_brazos"]["Mano Izquierda"] = {
                         "0_1": calcular_pendiente(
                             (results.left_hand_landmarks.landmark[0].x, results.left_hand_landmarks.landmark[0].y),
                             (results.left_hand_landmarks.landmark[1].x, results.left_hand_landmarks.landmark[1].y)
                         ),
                         "1_2": calcular_pendiente(
                             (results.left_hand_landmarks.landmark[1].x, results.left_hand_landmarks.landmark[1].y),
                             (results.left_hand_landmarks.landmark[2].x, results.left_hand_landmarks.landmark[2].y)
                         ),
                         "2_3": calcular_pendiente(
                              (results.left_hand_landmarks.landmark[2].x, results.left_hand_landmarks.landmark[2].y),
                              (results.left_hand_landmarks.landmark[3].x, results.left_hand_landmarks.landmark[3].y)
                         ),
                         "3_4": calcular_pendiente(
                              (results.left_hand_landmarks.landmark[3].x, results.left_hand_landmarks.landmark[3].y),
                              (results.left_hand_landmarks.landmark[4].x, results.left_hand_landmarks.landmark[4].y)
                         ),
                         "0_5": calcular_pendiente(
                              (results.left_hand_landmarks.landmark[0].x, results.left_hand_landmarks.landmark[0].y),
                              (results.left_hand_landmarks.landmark[5].x, results.left_hand_landmarks.landmark[5].y)
                         ),
                         "5_6": calcular_pendiente(
                              (results.left_hand_landmarks.landmark[5].x, results.left_hand_landmarks.landmark[5].y),
                              (results.left_hand_landmarks.landmark[6].x, results.left_hand_landmarks.landmark[6].y)
                         ),
                         "6_7": calcular_pendiente(
                              (results.left_hand_landmarks.landmark[6].x, results.left_hand_landmarks.landmark[6].y),
                              (results.left_hand_landmarks.landmark[7].x, results.left_hand_landmarks.landmark[7].y)
                         ),
                         "7_8": calcular_pendiente(
                              (results.left_hand_landmarks.landmark[7].x, results.left_hand_landmarks.landmark[7].y),
                              (results.left_hand_landmarks.landmark[8].x, results.left_hand_landmarks.landmark[8].y)
                         ),
                         "5_9": calcular_pendiente(
                              (results.left_hand_landmarks.landmark[5].x, results.left_hand_landmarks.landmark[5].y),
                              (results.left_hand_landmarks.landmark[9].x, results.left_hand_landmarks.landmark[9].y)
                         ),
                         "9_10": calcular_pendiente(
                              (results.left_hand_landmarks.landmark[9].x, results.left_hand_landmarks.landmark[9].y),
                              (results.left_hand_landmarks.landmark[10].x, results.left_hand_landmarks.landmark[10].y)
                         ),
                         "10_11": calcular_pendiente(
                              (results.left_hand_landmarks.landmark[10].x, results.left_hand_landmarks.landmark[10].y),
                              (results.left_hand_landmarks.landmark[11].x, results.left_hand_landmarks.landmark[11].y)
                         ),
                         "11_12": calcular_pendiente(
                              (results.left_hand_landmarks.landmark[11].x, results.left_hand_landmarks.landmark[11].y),
                              (results.left_hand_landmarks.landmark[12].x, results.left_hand_landmarks.landmark[12].y)
                         ),
                         "9_13": calcular_pendiente(
                              (results.left_hand_landmarks.landmark[9].x, results.left_hand_landmarks.landmark[9].y),
                              (results.left_hand_landmarks.landmark[13].x, results.left_hand_landmarks.landmark[13].y)
                         ),
                         "13_14": calcular_pendiente(
                              (results.left_hand_landmarks.landmark[13].x, results.left_hand_landmarks.landmark[13].y),
                              (results.left_hand_landmarks.landmark[14].x, results.left_hand_landmarks.landmark[14].y)
                         ),
                         "14_15": calcular_pendiente(
                              (results.left_hand_landmarks.landmark[14].x, results.left_hand_landmarks.landmark[14].y),
                              (results.left_hand_landmarks.landmark[15].x, results.left_hand_landmarks.landmark[15].y)
                         ),
                         "15_16": calcular_pendiente(
                              (results.left_hand_landmarks.landmark[15].x, results.left_hand_landmarks.landmark[15].y),
                              (results.left_hand_landmarks.landmark[16].x, results.left_hand_landmarks.landmark[16].y)
                         ),
                         "13_17": calcular_pendiente(
                              (results.left_hand_landmarks.landmark[13].x, results.left_hand_landmarks.landmark[13].y),
                              (results.left_hand_landmarks.landmark[17].x, results.left_hand_landmarks.landmark[17].y)
                         ),
                         "0_17": calcular_pendiente(
                              (results.left_hand_landmarks.landmark[0].x, results.left_hand_landmarks.landmark[0].y),
                              (results.left_hand_landmarks.landmark[17].x, results.left_hand_landmarks.landmark[17].y)
                         ),
                         "17_18": calcular_pendiente(
                              (results.left_hand_landmarks.landmark[17].x, results.left_hand_landmarks.landmark[17].y),
                              (results.left_hand_landmarks.landmark[18].x, results.left_hand_landmarks.landmark[18].y)
                         ),
                         "18_19": calcular_pendiente(
                              (results.left_hand_landmarks.landmark[18].x, results.left_hand_landmarks.landmark[18].y),
                              (results.left_hand_landmarks.landmark[19].x, results.left_hand_landmarks.landmark[19].y)
                         ),
                         "19_20": calcular_pendiente(
                              (results.left_hand_landmarks.landmark[19].x, results.left_hand_landmarks.landmark[19].y),
                              (results.left_hand_landmarks.landmark[20].x, results.left_hand_landmarks.landmark[20].y)
                         )
                    }
               else:
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

               # Mostrar el frame
               cv2.imshow("Frame", frame)

               # Agregar los datos del fotograma a la lista
               frames_data.append(frame_data)

               # Actualizar el tiempo del último fotograma capturado
               last_capture_time = current_time

          # Verificar si han pasado 5 segundos
          if time.time() - start_time > 10:
               break

          # Salir si se presiona la tecla ESC
          if cv2.waitKey(1) & 0xFF == 27:
               break

# Liberar la cámara y cerrar las ventanas
cap.release()
cv2.destroyAllWindows()

# Guardar los datos en un archivo JSON
with open("A.json", "w") as f:
    json.dump(frames_data, f, indent=4)

print("Datos guardados en frames_data_nueva_estructura.json")
print(f"Total de fotogramas capturados: {len(frames_data)}")