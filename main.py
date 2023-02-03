# Universidad del Valle de Guatemala
# Facultad de Ingeniería
# Departamento de Ciencias de la Computación
# Minería de Datos

# Christopher García 20541
# Ma. Isabel Solano 20504

# impotar librarías

import numpy as np
import matplotlib as plt
import pandas as pd


# importar el conjunto de datos
datos = pd.read_csv('baseball_reference_2016_scrape.csv')
X = datos.iloc[:, :-1].values
y = datos.iloc[:, -1].values

print("\nDatos head")
print(datos.head())

print("\nX")
# print(X)

print("\ny")
print(y)

print("\ndatos.ilos[:, 0]")
print(datos.iloc[:, 0])

# limpieza columna por columna -------

# Attendance: remove "']"
for x in range(len(datos.iloc[:, 0])):
    string_temp = datos.iloc[:, 0][x][:-2]
    string_temp = string_temp.replace(",", "")
    int_temp = int(string_temp)
    datos.iloc[:, 0][x] = int_temp
    

for x in datos.iloc[:, 0]:
    print(x)

