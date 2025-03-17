class PreProcesamiento:
    json_P = {
        "id": 1,
        "datos_brazos": {
            "Brazo Derecho": {
                "Brazo": 0,
                "Antebrazo": 0
            },
            "Brazo Izquierdo": {
                "Brazo": 0,
                "Antebrazo": 0
            },
            "Mano Derecha": {
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
                "19_20": 0
            },
            "Mano Izquierda": {
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
                "19_20": 0
            },
            "variable_Derecha": {
                "pulgar": 0,
                "indice": 0,
                "medio": 0,
                "anular": 0,
                "menique": 0,
                "0_X": 0,
                "0_Y": 0,
                "4_X": 0,
                "4_Y": 0,
                "8_X": 0,
                "8_Y": 0,
                "12_X": 0,
                "12_Y": 0,
                "16_X": 0,
                "16_Y": 0,
                "20_X": 0,
                "20_Y": 0
            },
            "variable_Izquierda": {
                "pulgar": 0,
                "indice": 0,
                "medio": 0,
                "anular": 0,
                "menique": 0,
                "0_X": 0,
                "0_Y": 0,
                "4_X": 0,
                "4_Y": 0,
                "8_X": 0,
                "8_Y": 0,
                "12_X": 0,
                "12_Y": 0,
                "16_X": 0,
                "16_Y": 0,
                "20_X": 0,
                "20_Y": 0
            }
        }
    }
    def __init__(self):
        pass

    def movimiento(self,json_A):
        movD = []
        movI = []
        datos_A = json_A["datos_brazos"]
        datos_P = self.json_P["datos_brazos"]

        for mano in ["Mano Derecha"]:
            claves_ordenadas = sorted(datos_A[mano].keys())
            for clave in claves_ordenadas:
                #print(str(datos_A[mano][clave]) + " - " + str(datos_P[mano][clave]))
                movD.append(datos_A[mano][clave] - datos_P[mano][clave])

        for mano in ["Mano Izquierda"]:
            claves_ordenadas = sorted(datos_A[mano].keys())
            for clave in claves_ordenadas:
                movI.append(datos_A[mano][clave] - datos_P[mano][clave])

        for d in movD:
            if(abs(d) > 45):
                return "Hay movimiento: " + str(d)
        for i in movI:
            if(abs(i) > 45):
                return "Hay movimiento: " + str(i)
        return "No hay movimiento"

    def m(self, punto1, punto2): #Calcula la pendiente
        x1, y1 = punto1
        x2, y2 = punto2
        if x2 - x1 == 0:  # Evitar división por cero
            return float("inf")  # Pendiente infinita (línea vertical)
        return (y2 - y1) / (x2 - x1)
      
    def pendiente(self, results, frame_data, puntosDer, puntosIzq):
        # Postura (brazos)
                if results.pose_landmarks:
                    frame_data["datos_brazos"]["Brazo Derecho"] = {
                    "Brazo": self.m((results.pose_landmarks.landmark[11].x,results.pose_landmarks.landmark[11].y,),(results.pose_landmarks.landmark[13].x,results.pose_landmarks.landmark[13].y,),),
                    "Antebrazo": self.m((results.pose_landmarks.landmark[13].x,results.pose_landmarks.landmark[13].y,),(results.pose_landmarks.landmark[15].x,results.pose_landmarks.landmark[15].y,),),
                }
                    frame_data["datos_brazos"]["Brazo Izquierdo"] = {
                    "Brazo": self.m((results.pose_landmarks.landmark[12].x,results.pose_landmarks.landmark[12].y,),(results.pose_landmarks.landmark[14].x,results.pose_landmarks.landmark[14].y,),),
                    "Antebrazo": self.m((results.pose_landmarks.landmark[14].x,results.pose_landmarks.landmark[14].y,),(results.pose_landmarks.landmark[16].x,results.pose_landmarks.landmark[16].y,),),
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
                    "0_1": self.m((results.right_hand_landmarks.landmark[0].x,results.right_hand_landmarks.landmark[0].y,),(results.right_hand_landmarks.landmark[1].x, results.right_hand_landmarks.landmark[1].y,),),
                    "1_2": self.m((results.right_hand_landmarks.landmark[1].x,results.right_hand_landmarks.landmark[1].y,),(results.right_hand_landmarks.landmark[2].x,results.right_hand_landmarks.landmark[2].y,),),
                    "2_3": self.m((results.right_hand_landmarks.landmark[2].x, results.right_hand_landmarks.landmark[2].y,), ( results.right_hand_landmarks.landmark[3].x, results.right_hand_landmarks.landmark[3].y, ),),
                    "3_4": self.m((results.right_hand_landmarks.landmark[3].x, results.right_hand_landmarks.landmark[3].y,),(results.right_hand_landmarks.landmark[4].x,results.right_hand_landmarks.landmark[4].y,),),
                    "0_5": self.m( (results.right_hand_landmarks.landmark[0].x,results.right_hand_landmarks.landmark[0].y,),(results.right_hand_landmarks.landmark[5].x, results.right_hand_landmarks.landmark[5].y, ),),
                    "5_6": self.m((results.right_hand_landmarks.landmark[5].x,results.right_hand_landmarks.landmark[5].y,),(results.right_hand_landmarks.landmark[6].x,results.right_hand_landmarks.landmark[6].y,),),
                    "6_7": self.m((results.right_hand_landmarks.landmark[6].x,results.right_hand_landmarks.landmark[6].y, ),(results.right_hand_landmarks.landmark[7].x, results.right_hand_landmarks.landmark[7].y,),),
                    "7_8": self.m((results.right_hand_landmarks.landmark[7].x,results.right_hand_landmarks.landmark[7].y,), (results.right_hand_landmarks.landmark[8].x, results.right_hand_landmarks.landmark[8].y,),),
                    "5_9": self.m((results.right_hand_landmarks.landmark[5].x,results.right_hand_landmarks.landmark[5].y,),(results.right_hand_landmarks.landmark[9].x,results.right_hand_landmarks.landmark[9].y,),),
                    "9_10": self.m((results.right_hand_landmarks.landmark[9].x,results.right_hand_landmarks.landmark[9].y,),(results.right_hand_landmarks.landmark[10].x,results.right_hand_landmarks.landmark[10].y,),),
                    "10_11": self.m((results.right_hand_landmarks.landmark[10].x,results.right_hand_landmarks.landmark[10].y,), (results.right_hand_landmarks.landmark[11].x,results.right_hand_landmarks.landmark[11].y,), ),
                    "11_12": self.m((results.right_hand_landmarks.landmark[11].x,results.right_hand_landmarks.landmark[11].y,),(results.right_hand_landmarks.landmark[12].x,results.right_hand_landmarks.landmark[12].y,),),
                    "9_13": self.m((results.right_hand_landmarks.landmark[9].x,results.right_hand_landmarks.landmark[9].y,),(results.right_hand_landmarks.landmark[13].x,results.right_hand_landmarks.landmark[13].y,),),
                    "13_14": self.m((results.right_hand_landmarks.landmark[13].x,results.right_hand_landmarks.landmark[13].y,),(results.right_hand_landmarks.landmark[14].x,results.right_hand_landmarks.landmark[14].y,),),
                    "14_15": self.m((results.right_hand_landmarks.landmark[14].x,results.right_hand_landmarks.landmark[14].y,),(results.right_hand_landmarks.landmark[15].x,results.right_hand_landmarks.landmark[15].y,),),
                    "15_16": self.m((results.right_hand_landmarks.landmark[15].x,results.right_hand_landmarks.landmark[15].y,),(results.right_hand_landmarks.landmark[16].x,results.right_hand_landmarks.landmark[16].y,),),
                    "13_17": self.m((results.right_hand_landmarks.landmark[13].x,results.right_hand_landmarks.landmark[13].y,),(results.right_hand_landmarks.landmark[17].x,results.right_hand_landmarks.landmark[17].y,),),
                    "0_17": self.m((results.right_hand_landmarks.landmark[0].x,results.right_hand_landmarks.landmark[0].y,),(results.right_hand_landmarks.landmark[17].x,results.right_hand_landmarks.landmark[17].y,),),
                    "17_18": self.m((results.right_hand_landmarks.landmark[17].x,results.right_hand_landmarks.landmark[17].y,),(results.right_hand_landmarks.landmark[18].x,results.right_hand_landmarks.landmark[18].y,),),
                    "18_19": self.m((results.right_hand_landmarks.landmark[18].x,results.right_hand_landmarks.landmark[18].y,),(results.right_hand_landmarks.landmark[19].x, results.right_hand_landmarks.landmark[19].y,),),
                    "19_20": self.m((results.right_hand_landmarks.landmark[19].x,results.right_hand_landmarks.landmark[19].y,),(results.right_hand_landmarks.landmark[20].x,results.right_hand_landmarks.landmark[20].y,),),
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
                    "0_1": self.m((results.left_hand_landmarks.landmark[0].x,results.left_hand_landmarks.landmark[0].y,),(results.left_hand_landmarks.landmark[1].x,results.left_hand_landmarks.landmark[1].y,),),
                    "1_2": self.m((results.left_hand_landmarks.landmark[1].x,results.left_hand_landmarks.landmark[1].y,),(results.left_hand_landmarks.landmark[2].x,results.left_hand_landmarks.landmark[2].y,),),
                    "2_3": self.m((results.left_hand_landmarks.landmark[2].x,results.left_hand_landmarks.landmark[2].y,),(results.left_hand_landmarks.landmark[3].x,results.left_hand_landmarks.landmark[3].y,),),
                    "3_4": self.m((results.left_hand_landmarks.landmark[3].x,results.left_hand_landmarks.landmark[3].y,),(results.left_hand_landmarks.landmark[4].x,results.left_hand_landmarks.landmark[4].y,),),
                    "0_5": self.m((results.left_hand_landmarks.landmark[0].x,results.left_hand_landmarks.landmark[0].y,),(results.left_hand_landmarks.landmark[5].x,results.left_hand_landmarks.landmark[5].y,),),
                    "5_6": self.m((results.left_hand_landmarks.landmark[5].x,results.left_hand_landmarks.landmark[5].y,),(results.left_hand_landmarks.landmark[6].x,results.left_hand_landmarks.landmark[6].y,),),
                    "6_7": self.m((results.left_hand_landmarks.landmark[6].x,results.left_hand_landmarks.landmark[6].y,),(results.left_hand_landmarks.landmark[7].x,results.left_hand_landmarks.landmark[7].y,),),
                    "7_8": self.m((results.left_hand_landmarks.landmark[7].x,results.left_hand_landmarks.landmark[7].y,),(results.left_hand_landmarks.landmark[8].x,results.left_hand_landmarks.landmark[8].y,),),
                    "5_9": self.m((results.left_hand_landmarks.landmark[5].x,results.left_hand_landmarks.landmark[5].y,),(results.left_hand_landmarks.landmark[9].x,results.left_hand_landmarks.landmark[9].y,),),
                    "9_10": self.m((results.left_hand_landmarks.landmark[9].x,results.left_hand_landmarks.landmark[9].y,),(results.left_hand_landmarks.landmark[10].x,results.left_hand_landmarks.landmark[10].y,),),
                    "10_11": self.m((results.left_hand_landmarks.landmark[10].x,results.left_hand_landmarks.landmark[10].y,),(results.left_hand_landmarks.landmark[11].x,results.left_hand_landmarks.landmark[11].y,),),
                    "11_12": self.m((results.left_hand_landmarks.landmark[11].x,results.left_hand_landmarks.landmark[11].y,),(results.left_hand_landmarks.landmark[12].x,results.left_hand_landmarks.landmark[12].y,),),
                    "9_13": self.m((results.left_hand_landmarks.landmark[9].x,results.left_hand_landmarks.landmark[9].y,),(results.left_hand_landmarks.landmark[13].x,results.left_hand_landmarks.landmark[13].y,),),
                    "13_14": self.m((results.left_hand_landmarks.landmark[13].x,results.left_hand_landmarks.landmark[13].y,),(results.left_hand_landmarks.landmark[14].x,results.left_hand_landmarks.landmark[14].y,),),
                    "14_15": self.m((results.left_hand_landmarks.landmark[14].x,results.left_hand_landmarks.landmark[14].y,),(results.left_hand_landmarks.landmark[15].x,results.left_hand_landmarks.landmark[15].y,),),
                    "15_16": self.m((results.left_hand_landmarks.landmark[15].x,results.left_hand_landmarks.landmark[15].y,),(results.left_hand_landmarks.landmark[16].x,results.left_hand_landmarks.landmark[16].y,),),
                    "13_17": self.m((results.left_hand_landmarks.landmark[13].x,results.left_hand_landmarks.landmark[13].y,),(results.left_hand_landmarks.landmark[17].x,results.left_hand_landmarks.landmark[17].y,),),
                    "0_17": self.m((results.left_hand_landmarks.landmark[0].x,results.left_hand_landmarks.landmark[0].y,),(results.left_hand_landmarks.landmark[17].x,results.left_hand_landmarks.landmark[17].y,),),
                    "17_18": self.m((results.left_hand_landmarks.landmark[17].x,results.left_hand_landmarks.landmark[17].y,),(results.left_hand_landmarks.landmark[18].x,results.left_hand_landmarks.landmark[18].y,),),
                    "18_19": self.m((results.left_hand_landmarks.landmark[18].x,results.left_hand_landmarks.landmark[18].y,),(results.left_hand_landmarks.landmark[19].x,results.left_hand_landmarks.landmark[19].y,),),
                    "19_20": self.m((results.left_hand_landmarks.landmark[19].x,results.left_hand_landmarks.landmark[19].y,),(results.left_hand_landmarks.landmark[20].x,results.left_hand_landmarks.landmark[20].y,),),
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
                return frame_data, puntosDer, puntosIzq

    def desplazamiento(self, puntos1, puntos2):
        des = []
        for i in enumerate(puntos1):
            x = puntos2[i[0]][0] - puntos1[i[0]][0]
            y = puntos2[i[0]][1] - puntos1[i[0]][1]
            des.append([x, y])
        return des

    def variacion(self, jason, puntosDer, puntosIzq, tempPuntosDer, tempPuntosIzq):
        jason ["datos_brazos"]["variable_Derecha"] = {
            "0_X": self.desplazamiento(tempPuntosDer, puntosDer)[0][0],
            "0_Y": self.desplazamiento(tempPuntosDer, puntosDer)[0][1],
            "4_X": self.desplazamiento(tempPuntosDer, puntosDer)[1][0],
            "4_Y": self.desplazamiento(tempPuntosDer, puntosDer)[1][1],
            "8_X": self.desplazamiento(tempPuntosDer, puntosDer)[2][0],
            "8_Y": self.desplazamiento(tempPuntosDer, puntosDer)[2][1],
            "12_X": self.desplazamiento(tempPuntosDer, puntosDer)[3][0],
            "12_Y": self.desplazamiento(tempPuntosDer, puntosDer)[3][1],
            "16_X": self.desplazamiento(tempPuntosDer, puntosDer)[4][0],
            "16_Y": self.desplazamiento(tempPuntosDer, puntosDer)[4][1],
            "20_X": self.desplazamiento(tempPuntosDer, puntosDer)[5][0],
            "20_Y": self.desplazamiento(tempPuntosDer, puntosDer)[5][1]
        
        }

        jason ["datos_brazos"]["variable_Izquierda"] = {
            "0_X": self.desplazamiento(tempPuntosIzq, puntosIzq)[0][0],
            "0_Y": self.desplazamiento(tempPuntosIzq, puntosIzq)[0][1],
            "4_X": self.desplazamiento(tempPuntosIzq, puntosIzq)[1][0],
            "4_Y": self.desplazamiento(tempPuntosIzq, puntosIzq)[1][1],
            "8_X": self.desplazamiento(tempPuntosIzq, puntosIzq)[2][0],
            "8_Y": self.desplazamiento(tempPuntosIzq, puntosIzq)[2][1],
            "12_X": self.desplazamiento(tempPuntosIzq, puntosIzq)[3][0],
            "12_Y": self.desplazamiento(tempPuntosIzq, puntosIzq)[3][1],
            "16_X": self.desplazamiento(tempPuntosIzq, puntosIzq)[4][0],
            "16_Y": self.desplazamiento(tempPuntosIzq, puntosIzq)[4][1],
            "20_X": self.desplazamiento(tempPuntosIzq, puntosIzq)[5][0],
            "20_Y": self.desplazamiento(tempPuntosIzq, puntosIzq)[5][1]

        }
        return jason

    def setJson_P(self, jason):
        self.json_P = jason
    
    def ReiniciarJson(self):
        self.json_P = {
        "id": 1,
        "datos_brazos": {
            "Brazo Derecho": {
                "Brazo": 0,
                "Antebrazo": 0
            },
            "Brazo Izquierdo": {
                "Brazo": 0,
                "Antebrazo": 0
            },
            "Mano Derecha": {
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
                "19_20": 0
            },
            "Mano Izquierda": {
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
                "19_20": 0
            },
            "variable_Derecha": {
                "pulgar": 0,
                "indice": 0,
                "medio": 0,
                "anular": 0,
                "menique": 0,
                "0_X": 0,
                "0_Y": 0,
                "4_X": 0,
                "4_Y": 0,
                "8_X": 0,
                "8_Y": 0,
                "12_X": 0,
                "12_Y": 0,
                "16_X": 0,
                "16_Y": 0,
                "20_X": 0,
                "20_Y": 0
            },
            "variable_Izquierda": {
                "pulgar": 0,
                "indice": 0,
                "medio": 0,
                "anular": 0,
                "menique": 0,
                "0_X": 0,
                "0_Y": 0,
                "4_X": 0,
                "4_Y": 0,
                "8_X": 0,
                "8_Y": 0,
                "12_X": 0,
                "12_Y": 0,
                "16_X": 0,
                "16_Y": 0,
                "20_X": 0,
                "20_Y": 0
            }
        }
    }