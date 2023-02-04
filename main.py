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

datos = datos.loc[:, datos.columns!="field_type"]
datos.select_dtypes(exclude='object')

# arreglar columnas antes de guardar los datos
columns_to_drop = ["boxscore_url", "field_type", "other_info_string"]
datos = clean(datos, method = 'dropcols', columns = columns_to_drop)

datos = clean(datos, method = "standardize")
datos = clean(datos, method = "duplicates")
datos = clean(datos, method = "replaceval", columns = [], to_replace = "",  value = np.nan)
datos = clean(datos, method = "fillmissing")
datos = clean(datos, method = "dropmissing")
datos = clean(datos, method = "outliers", columns=[])

datos = datos[datos["attendance"].str.contains("Citi Field") == False]
datos = datos[datos["attendance"].str.contains("PNC Park") == False]
datos = datos[datos["attendance"].str.contains("U.S. Cellular Field") == False]
datos = datos[datos["game_duration"].str.contains("Day Game, on grass") == False]

X = datos.iloc[:, 1:]
y = datos.iloc[:, :1].values

print("\nDatos head")
# print(datos.head())

print(X)

print("\nX")
# print(X)

print("\ny")
print(y)

# print("\ndatos.iloc[:, 0]")
# print(X[:, 0])

# # limpieza columna por columna -------


# game duration: remove ": " at the beginning


X['game_duration'] = X['game_duration'].str.replace(": ", "")


# Venue: remove ":" at the beginning

X['venue'] = X['venue'].str.replace(": ", "")


# # start_time: remove unnecesary text 

X['start_time'] = X['start_time'].str.replace("Start Time: ", "")
X['start_time'] = X['start_time'].str.replace(" Local", "")


# # date: formatting

# X['date'] = X['date'].replace(X['date'] , pd.to_datetime(X['date'], format="%A, %B %d, %Y"))  

# # Impresión de la información de todas las columnas una por una. 

for x in X:
    # print("\nX[:, %d]"%x)
    # print(X[:, x])
    print(X[x])

# # Fixed y

for x in range(len(y)):
    string_temp = y[x][0][:-2].replace(",", "")
    int_temp = int(string_temp)
    y[x][0] = int_temp
    
print("fixed y")
print(y)


# # Codificación de los datos -------
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
transfCol = ColumnTransformer(transformers = [('encoder', OneHotEncoder(), [0])], remainder = 'passthrough')
X = np.array(transfCol.fit_transform(X))

# print(X)

def codif_y_ligar(dataframe_original, variables_a_codificar):
    dummies = pd.get_dummies(dataframe_original[[variables_a_codificar]])
    res = pd.concat([dataframe_original, dummies], axis = 1)
    res = res.drop([variables_a_codificar], axis = 1)
    return(res) 

# variables_a_codificar = [
#     'date',
#     # 'game_duration', 'game_type', 'home_team', 
#     # 'start_time', 'venue'
#     ]   #  Esta es una lista de variables
# for variable in variables_a_codificar:
#     X = codif_y_ligar(X, variable)

    
# división de los datos ----

from sklearn.model_selection import train_test_split
seed = 0 # puede ser aleatoria
X_entreno, X_prueba, y_entreno, y_prueba = train_test_split(X, y, test_size = 1/3, random_state = seed)


# Entrenamiento del modelo de regresión lineal simple
from sklearn.linear_model import LinearRegression
regresor = LinearRegression()
# regresor.fit(X_entreno, y_entreno)