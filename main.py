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
from pandas_profiling import ProfileReport
import matplotlib.pyplot as plt
import seaborn as sns

from quickda.explore_data import *
from quickda.clean_data import *
from quickda.explore_numeric import *
from quickda.explore_categoric import *
from quickda.explore_numeric_categoric import *
from quickda.explore_time_series import *

# importar el conjunto de datos
datos = pd.read_csv('baseball_reference_2016_scrape.csv')

# arreglar columnas antes de guardar los datos
columns_to_drop = ["boxscore_url", "field_type"]
datos = clean(datos, method = 'dropcols', columns = columns_to_drop)

datos = clean(datos, method = "standardize")
datos = clean(datos, method = "duplicates")
datos = clean(datos, method = "replaceval", columns = [], to_replace = "",  value = np.nan)
datos = clean(datos, method = "fillmissing")
datos = clean(datos, method = "dropmissing")
datos = clean(datos, method = "outliers", columns=[])

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


# print("\nprint(X[:, 0])")
# print(X[:, 0])


# game duration: remove ": " at the beginning

for x in range(len(X[:, 6])):
    string_temp = X[:, 6][x].replace(": ", "")
    X[:, 6][x] = string_temp

# print("\nprint(X[:, 8])")
# print(X[:, 8])


# Venue: remove ":" at the beginning

for x in range(len(X[:, 13])):
    string_temp = X[:, 13][x].replace(": ", "")
    X[:, 13][x] = string_temp

# print("\nprint(X[:, 16])")
# print(X[:, 16])


# start_time: remove unnecesary text

for x in range(len(X[:, 12])):
    string_temp = X[:, 12][x].replace("Start Time: ", "").replace(" Local", "")
    X[:, 12][x] = string_temp

# print("\nprint(X[:, 15])")
# print(X[:, 15])


# date: formattiog

for x in range(len(X[:, 5])):
    date_temp = pd.to_datetime(X[:, 5][x], format="%A, %B %d, %Y")
    X[:, 5][x] = date_temp

# print("\nprint(X[:, 6])")
# print(X[:, 6])

for x in range(14):
    print("\nX[:, %d]"%x)
    print(X[:, x])

# Fixed y

for x in range(len(y)):
    string_temp = y[x][0][:-2].replace(",", "")
    int_temp = int(string_temp)
    y[x][0] = int_temp
    # print(y[x][0])
    
print(y)
    
# división de los datos ----

from sklearn.model_selection import train_test_split
seed = 0 # puede ser aleatoria
X_entreno, X_prueba, y_entreno, y_prueba = train_test_split(X, y, test_size = 1/3, random_state = seed)


# Entrenamiento del modelo de regresión lineal simple
from sklearn.linear_model import LinearRegression
regresor = LinearRegression()
# regresor.fit(X_entreno, y_entreno)

