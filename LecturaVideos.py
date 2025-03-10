import os

class LecturaVideos:

    def __init__(self):
        pass

    def extraccion(self, ruta):
        rutas = []
        for carpeta in os.listdir(ruta):
            ruta_Carpeta = os.path.join(ruta, carpeta)
            if os.path.isdir(ruta_Carpeta):
                for archivo in os.listdir(ruta_Carpeta):
                    if archivo.endswith(".avi"):
                        ruta_archivo = os.path.join(ruta_Carpeta, archivo)
                        #print("La ruta es:" + ruta_archivo)
                        #print("Clase:" + carpeta)
                        rutas.append([ruta_archivo, carpeta])
        return rutas
        