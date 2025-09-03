# -*- coding: utf-8 -*-
"""
Created on Tue Sep  2 22:01:42 2025

@author: qbo28
"""

# task_01_preprocessing.py
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def load_and_preprocess_data():
    # 🔹 Ruta absoluta del archivo iris.csv
    path = r"C:\Users\qbo28\OneDrive\Documentos\CVs\Codveda_Internship\Data Set For Task-20250902T054136Z-1-001\Data Set For Task\1) iris.csv"
    
    # 1. Cargar dataset
    df = pd.read_csv(path)

    # 2. Revisar columnas
    print("Columnas detectadas:", df.columns.tolist())

    # 3. Normalización / estandarización de columnas numéricas
    scaler = StandardScaler()
    df[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']] = scaler.fit_transform(
        df[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
    )

    # 🔹 Aquí dejamos species como etiqueta de clasificación (3 clases)
    X = df.drop(columns=['species'])
    y = df['species']

    # 4. Separación train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    return X_train, X_test, y_train, y_test

