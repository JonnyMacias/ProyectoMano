import csv
import pandas as pd

# Leer el CSV
with open('corpues.csv', 'r', encoding='utf-8') as file:
    reader = csv.reader(file)
    data = list(reader)

# Encontrar la fila con mayor cantidad de columnas
max_cols = max(len(row) for row in data)

# Asegurar que la última columna permanezca fija
def ajustar_fila(row, max_cols):
    if len(row) == 0:
        return ['0'] * max_cols
    last_value = row[-1]  # Último valor
    row = row[:-1]  # Todas menos la última columna
    # Rellenar con ceros y agregar la última columna al final
    return row + ['0'] * (max_cols - len(row) - 1) + [last_value]

# Ajustar todas las filas
data_padded = [ajustar_fila(row, max_cols) for row in data]

# Convertir a DataFrame
df = pd.DataFrame(data_padded)

# Guardar el resultado
df.to_csv('N_archivo.csv', index=False, header=False)
print(f"Archivo normalizado con {max_cols} columnas, manteniendo la última columna fija.")
