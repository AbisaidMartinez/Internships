# -*- coding: utf-8 -*-
"""
Created on Tue Sep  2 00:04:12 2025

@author: Abisaid Martinez
"""

import pandas as pd

df = pd.read_csv(r'C:\Users\qbo28\OneDrive\Documentos\CVs\Codveda_Internship\Data Set For Task-20250902T054136Z-1-001\Data Set For Task\1) iris.csv')

print("Columns in the dataset:")
print(df.columns.tolist())


print("First rows in the dataset:")
print(df.head())
print("\nGeneral info:")
print(df.info())

print("Missing Values?")
print(df.isnull().sum())

print(df.describe(include='all'))

df_encoded = pd.get_dummies(df, columns=['species'], drop_first=True)
print("Columns after encoding:", df_encoded.columns.tolist())

from sklearn.preprocessing import StandardScaler

# Codificación de la variable categórica "species"
df_encoded = pd.get_dummies(df, columns=['species'], drop_first=True)

scaler = StandardScaler()
df_encoded[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']] = scaler.fit_transform(
    df_encoded[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
)

print(df_encoded.head())

from sklearn.model_selection import train_test_split

X = df_encoded.drop(columns=['species_versicolor', 'species_virginica'])
y = df_encoded[['species_versicolor', 'species_virginica']]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"Train size: {X_train.shape[0]} samples")
print(f"Test size: {X_test.shape[0]} samples")
