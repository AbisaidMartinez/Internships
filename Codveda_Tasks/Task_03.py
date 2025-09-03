# -*- coding: utf-8 -*-
"""
Created on Wed Sep  3 22:46:28 2025

@author: Abisaid Martinez
"""


import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, classification_report
from task_01_preprocessing import load_and_preprocess_data

# 1. Load processed data from Task 1
X_train, X_test, y_train, y_test = load_and_preprocess_data()

# 2. Train Random Forest model with basic hyperparameters
rf = RandomForestClassifier(
    n_estimators=200,   # númber of trees
    criterion = "entropy",
    max_depth=20,        # Max depth of each tree
    random_state=42
)
rf.fit(X_train, y_train)

# 3. Predictions
y_pred = rf.predict(X_test)

# 4. Evaluation in testset
print("✅ Accuracy:", accuracy_score(y_test, y_pred))
print("\n🔎 Full Report:\n", classification_report(y_test, y_pred))

# 5. Cross-validation in train set
cv_scores = cross_val_score(rf, X_train, y_train, cv=5, scoring='f1_macro')
print("\n📊 F1-score (cross-validation, 5 folds):", cv_scores)
print("Mean:", cv_scores.mean())

# 6. Features Importance
feature_importances = pd.Series(rf.feature_importances_, index=X_train.columns).sort_values(ascending=False)

print("\n🌟 Features Importance:\n", feature_importances)

# 7. Feature Importance Visualization
plt.figure(figsize=(8,5))
ax = feature_importances.plot(kind='bar', color="skyblue", edgecolor="black")
plt.title("Importancia de las características - Random Forest")
plt.ylabel("Importancia")
plt.xlabel("Características")

# Add values  encima de cada barra
for i, v in enumerate(feature_importances):
    ax.text(i, v + 0.01, f"{v:.2f}", ha='center', va='bottom', fontsize=9, fontweight="bold")

plt.tight_layout()
plt.show()
