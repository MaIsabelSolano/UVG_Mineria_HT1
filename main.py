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

# arreglar columnas antes de guardar los datos


X = datos.iloc[:, :].values
y = datos.iloc[:, :1].values

print("\nDatos head")
print(datos.head())

print("\nX")
# print(X)

print("\ny")
print(y)

print("\ndatos.iloc[:, 0]")
print(X[:, 0])

# limpieza columna por columna -------


# Attendance: remove "']" at the end

for x in range(len(X[:, 0])):
    string_temp = X[:, 0][x][:-2]
    string_temp = string_temp.replace(",", "")
    int_temp = int(string_temp)
    X[:, 0][x] = int_temp


print("\nprint(X[:, 0])")
print(X[:, 0])


print("\nprint(X[:, 0])")
print(X[:, 0])


# game duration: remove ": " at the beginning

for x in range(len(X[:, 8])):
    string_temp = X[:, 8][x].replace(": ", "")
    X[:, 8][x] = string_temp

print("\nprint(X[:, 8])")
print(X[:, 8])


# Venue: remove ":" at the beginning

for x in range(len(X[:, 16])):
    string_temp = X[:, 16][x].replace(": ", "")
    X[:, 16][x] = string_temp

print("\nprint(X[:, 16])")
print(X[:, 16])


# start_time: remove unnecesary text

for x in range(len(X[:, 15])):
    string_temp = X[:, 15][x].replace("Start Time: ", "").replace(" Local", "")
    X[:, 15][x] = string_temp

print("\nprint(X[:, 15])")
print(X[:, 15])


# date: formattiog

for x in range(len(X[:, 6])):
    date_temp = pd.to_datetime(X[:, 6][x], format="%A, %B %d, %Y")
    X[:, 6][x] = date_temp

print("\nprint(X[:, 6])")
print(X[:, 6])


for x in range(17):
    print("\nX[:, %d]"%x)
    print(X[:, x])

# Fixed y

for x in range(len(y)):
    string_temp = y[x][0][:-2].replace(",", "")
    int_temp = int(string_temp)
    y[x][0] = int_temp
    # print(y[x][0])
    
print(y)
    

# normalización de datos ------


