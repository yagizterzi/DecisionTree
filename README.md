# Decision Tree and Random Forest Implementation

## Overview
This project implements decision tree and random forest algorithms from scratch using NumPy. The implementation provides flexible, customizable classifiers that can be used for various machine learning tasks.

## Classes

### DecisionTree
A decision tree classifier that uses the Gini impurity criterion for node splitting.

#### Features
- Customizable maximum tree depth
- Minimum samples required to split a node
- Minimum impurity decrease threshold for splits
- Feature importance calculation
- Accuracy evaluation

#### Parameters
- `max_depth`: Maximum depth of the tree (default=5)
- `min_samples_split`: Minimum samples required to split a node (default=2)
- `min_impurity_decrease`: Minimum decrease in impurity required for split (default=0.0)

### RandomForest
An ensemble of decision trees that uses bootstrap sampling and random feature selection.

#### Features
- Multiple decision trees for robust predictions
- Bootstrap sampling of training data
- Random feature selection for diversity
- Majority voting for final predictions

#### Parameters
- `n_estimators`: Number of trees in the forest (default=10)
- `max_features`: Number of features to consider for best split (default="sqrt")
- `max_depth`: Maximum depth of each tree (default=5)
- `min_samples_split`: Minimum samples required to split a node (default=2)
- `min_impurity_decrease`: Minimum decrease in impurity required for split (default=0.0)
- `random_state`: Random seed for reproducibility (default=None)

## Implementation Details

- Uses dictionaries to represent tree nodes instead of formal node classes
- Implements Gini impurity criterion for splitting
- Calculates feature importance based on impurity decrease
- Employs recursive tree building and prediction

## Usage Examples

### Decision Tree

```python
from tree import DecisionTree
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# Load and prepare data
iris = load_iris()
X, y = iris.data, iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the decision tree
dt = DecisionTree(max_depth=5, min_samples_split=2)
dt.fit(X_train, y_train)

# Make predictions
y_pred = dt.predict(X_test)

# Calculate accuracy
accuracy = dt.accuracy(y_pred, y_test)
print(f"Accuracy: {accuracy:.4f}")

# Get feature importances
importances = dt.feature_importances()
for i, importance in enumerate(importances):
    print(f"Feature {i}: {importance:.4f}")
```

### Random Forest

```python
from tree import RandomForest

# Create and train the random forest
rf = RandomForest(n_estimators=10, max_depth=5, random_state=42)
rf.fit(X_train, y_train)

# Make predictions
y_pred_rf = rf.predict(X_test)

# Calculate accuracy
accuracy_rf = sum(y_pred_rf == y_test) / len(y_test)
print(f"Random Forest Accuracy: {accuracy_rf:.4f}")
```

## Requirements
- NumPy
- Optional: scikit-learn (for dataset loading and evaluation)

## Notes
- This implementation is designed for educational purposes
- The code prioritizes readability and transparency over optimization
- Suitable for small to medium-sized datasets
