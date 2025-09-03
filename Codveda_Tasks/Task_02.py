# -*- coding: utf-8 -*-
"""
Created on Wed Sep  3 17:31:29 2025

@author: Abisaid Martinez 
"""

import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, f1_score, classification_report
from task_01_preprocessing import load_and_preprocess_data

# 1. Load processed data from Task 1
X_train, X_test, y_train, y_test = load_and_preprocess_data()

# 2. Train Decision Tree
clf = DecisionTreeClassifier(criterion="log_loss", splitter='best', max_depth=3, random_state=42)  # pruning con max_depth
clf.fit(X_train, y_train)

# 3. Predictions
y_pred = clf.predict(X_test)

# 4. Evaluation
print("✅ Accuracy:", accuracy_score(y_test, y_pred))
print("✅ F1-score (macro):", f1_score(y_test, y_pred, average="macro"))
print("\n🔎 Full Report:\n", classification_report(y_test, y_pred))
print("Train Accuracy:", clf.score(X_train, y_train))
print("Test Accuracy:", clf.score(X_test, y_test))

# 5. Decision Tree
plt.figure(figsize=(12, 8))
plot_tree(
    clf,
    filled=True,
    feature_names=X_train.columns,
    class_names=clf.classes_
)
plt.title("Decision Tree - Iris")
plt.show()









# from sklearn.tree import DecisionTreeClassifier
# from sklearn.model_selection import train_test_split
# from task_01_preprocessing import load_and_preprocess_data

# # Supongamos que ya tienes X (features) y y (target)
# #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Entrenar el modelo
# clf = DecisionTreeClassifier(random_state=42)
# clf.fit(X_train, y_train)

# from sklearn.tree import plot_tree
# import matplotlib.pyplot as plt

# plt.figure(figsize=(12, 8))
# plot_tree(clf, feature_names=X.columns, class_names=clf.classes_, filled=True)
# plt.title("Árbol de Decisión - Iris")
# plt.show()

# clf_pruned = DecisionTreeClassifier(max_depth=3, random_state=42)
# clf_pruned.fit(X_train, y_train)

# from sklearn.metrics import accuracy_score, f1_score, classification_report

# y_pred = clf_pruned.predict(X_test)

# print("Accuracy:", accuracy_score(y_test, y_pred))
# print("F1-score (macro):", f1_score(y_test, y_pred, average='macro'))
# print("\nReporte de clasificación:\n", classification_report(y_test, y_pred))

