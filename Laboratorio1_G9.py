# Universidad del Valle de Guatemala
# Facultad de Ingeniería
# Departamento de Ciencias de la Computación
# Minería de Datos

# Christopher García 20541
# Ma. Isabel Solano 20504

# # Laboratorio_1

# ## Importar las librerías

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from pandas_profiling import ProfileReport
from quickda.explore_data import *
from quickda.clean_data import *
from quickda.explore_numeric import *
from quickda.explore_categoric import *
from quickda.explore_numeric_categoric import *
from quickda.explore_time_series import *
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


# ## Importar el conjunto de datos
datos = pd.read_csv("baseball_reference_2016_scrape.csv")

# ## Obtención de información sobre los datos
# print(datos.info()) # <- Se puede descomentar para ver impresión de datos

# ## Descripción adicional de los datos
profile = ProfileReport(datos)
# print(profile) # <- Se puede descomentar para ver impresión de datos

# ## Más descripción adicional de los datos
# print(datos.describe()) # <- Se puede descomentar para ver impresión de datos

# ## Obtención de datos numéricos
datos = datos.loc[:, datos.columns!="field_type"]
# print(datos.select_dtypes(exclude='object')) # <- Se puede descomentar para ver impresión de datos 

# ## Obtención de datos categóricos
# print(datos.select_dtypes(include='object'))  # <- Se puede descomentar para ver impresión de datos

# ## Obtención de datos para ver su estructura
datos.head()

'''

# ## Inicio gráficos de exploración
# ## Gráfico conteo de tipos de juego

sns.countplot(x = 'game_type', data = datos);

# ## Gráfico conteo de veces siendo locales por equipo

sns.countplot(y = 'home_team', data = datos);

# ## Gráfico conteo de veces siendo visitantes por equipo

sns.countplot(y = 'away_team', data = datos);

# ## Gráficos de correlación

sns.pairplot(datos, x_vars=['home_team_runs'], y_vars=['home_team_hits']);
sns.pairplot(datos, x_vars=['away_team_runs'], y_vars=['away_team_hits']);

# ## Gráficos de correlación (versión extendida)

sns.pairplot(datos);
sns.boxplot(data=datos, x='away_team_runs', y='home_team_runs')
sns.catplot(datos, x="home_team_errors", y="home_team_hits", kind="box")

# ## Gráficos de datos categóricos

sns.countplot(x = 'date', data = datos);
sns.countplot(y = 'venue', data = datos);
sns.countplot(y = 'boxscore_url', data = datos);

'''

# ## Proceso de limpieza
# explore(datos, method="summarize") 

# ## Eliminación de columnas
columns_to_drop = ["boxscore_url", "field_type", "other_info_string"]
datos = clean(datos, method = 'dropcols', columns = columns_to_drop)

# ## Estandarizar los nombres de las columnas
datos = clean(datos, method = "standardize")

# ## Eliminar filas duplicadas
datos = clean(datos, method = "duplicates")
datos = clean(datos, method = "replaceval", columns = [], to_replace = "",  value = np.nan)

# ## Reemplazar valores faltantes
datos = clean(datos, method = "fillmissing")

# ## Eliminar filas con valores faltantes
datos = clean(datos, method = "dropmissing")

# ## Eliminar datos atípicos
datos = clean(datos, method="outliers", columns=[])

# ## Arreglo columna Attendance, venue y game_duration
datos['attendance'] = datos['attendance'].str.replace("']", "")
datos['attendance'] = datos['attendance'].str.replace(",", "")
datos['game_duration'] = datos['game_duration'].str.replace(":", "")
datos['venue'] = datos['venue'].str.replace(":", "")
datos = datos[datos["attendance"].str.contains("Citi Field") == False]
datos = datos[datos["attendance"].str.contains("PNC Park") == False]
datos = datos[datos["attendance"].str.contains("U.S. Cellular Field") == False]
datos = datos[datos["game_duration"].str.contains("Day Game, on grass") == False]

# ## Cambiar tipo de datos
to_categoric = ["away_team", "date", "game_type", "home_team", "start_time", "venue"]
datos = clean(datos, method = 'dtypes', columns = to_categoric, dtype='category')

to_numeric = ["attendance", "game_duration"]
datos = clean(datos, method = 'dtypes', columns = to_numeric, dtype='int64')
datos["attendance"] = datos["attendance"].astype('int64')
datos["game_duration"] = datos["game_duration"].astype('int64')

# explore(datos, method="summarize") 

# ## Modelo de predicción

X = datos.iloc[:, 1:]
y = datos.iloc[:, 0]

# ## Codificación de datos categóricos

def codif_y_ligar(dataframe_original, variables_a_codificar):
    dummies = pd.get_dummies(dataframe_original[[variables_a_codificar]])
    res = pd.concat([dataframe_original, dummies], axis = 1)
    res = res.drop([variables_a_codificar], axis = 1)
    return(res) 

variables_a_codificar = ['away_team', 'date', 'game_type', 'home_team', 'start_time', 'venue']
for variable in variables_a_codificar:
    X = codif_y_ligar(X, variable)

# ## Datos de entreno y de prueba
X_entreno, X_prueba, y_entreno, y_prueba = train_test_split(X, y, test_size = 0.2, random_state = 1)
regresor = LinearRegression()
regresor.fit(X_entreno, y_entreno)

rCuadrado = regresor.score(X_entreno, y_entreno)
print("R² del modelo: ",rCuadrado)
rCuadradoCalculo = 1 - (1-regresor.score(X_entreno, y_entreno))*(len(y_entreno)-1)/(len(y_entreno)-X_entreno.shape[1]-1)
print("R² calculado: ",rCuadradoCalculo)
