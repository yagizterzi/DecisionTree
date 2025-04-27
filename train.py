import numpy as np
import pandas as pd
from tree import DecisionTree, RandomForest
from sklearn.model_selection import train_test_split
from sklearn import datasets

data = datasets.load_breast_cancer()
X, y = data.data, data.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=1234
)

dt = DecisionTree(max_depth=5, min_samples_split=2)
dt.fit(X_train, y_train)

y_pred = dt.predict(X_test)

accuracy = dt.accuracy(y_pred, y_test)
print(f"Accuracy: {accuracy:.4f}")


importances = dt.feature_importances()
for i, importance in enumerate(importances):
    print(f"Feature {i}: {importance:.4f}")






