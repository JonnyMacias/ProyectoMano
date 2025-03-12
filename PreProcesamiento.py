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
            
            
        

    def setJson_P(self, jason):
        self.json_P = jason