import numpy as np
# DecisionTree class implements a simple decision tree classifier.
class DecisionTree:
    def __init__(self, max_depth=5, min_samples_split=2, min_impurity_decrease=0.0):
        # Initialize the decision tree with given parameters.
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_impurity_decrease = min_impurity_decrease
        self.tree = None
        self.feature_importances_ = None
        self.class_mapping = None 

    # Fit the decision tree on the training data.
    def fit(self, X, y):
        self.n_features = X.shape[1]
        self.feature_importances_ = np.zeros(self.n_features)
        # Map each unique class to an integer index.
        self.class_mapping = {cls: idx for idx, cls in enumerate(np.unique(y))}
        # Build the tree recursively.
        self.tree = self._growtree(X, y)

    # Predict class labels for given input data.
    def predict(self, X):
        return np.array([self._predict(x, self.tree) for x in X])

    # Calculate the Gini impurity for an array of labels.
    def _gini(self, y):
        m = len(y)
        if m == 0:
            return 0
        return 1.0 - sum((np.sum(y == c) / m) ** 2 for c in np.unique(y))

    # Split the dataset based on a feature index and threshold.
    def _split(self, X, y, idx, t):
        left = np.where(X[:, idx] <= t)
        right = np.where(X[:, idx] > t)
        return (X[left], y[left]), (X[right], y[right])

    # Find the best split for the data by iterating over all features and thresholds.
    def _best_split(self, X, y):
        m, n = X.shape
        if m <= 1:
            return None, None

        num_parent = [np.sum(y == c) for c in self.class_mapping.keys()]
        best_gini = 1.0 - sum((i / m) ** 2 for i in num_parent)
        best_idx, best_t = None, None

        # Iterate over each feature.
        for idx in range(n):
            # Sort the values and corresponding classes for the current feature.
            threshold, classes = zip(*sorted(zip(X[:, idx], y)))
            num_left = [0] * len(num_parent)
            num_right = num_parent.copy()
            # Iterate through potential split points.
            for i in range(1, m):
                c_i = classes[i - 1]
                # Find the index of current class in class_mapping.
                # Using list(self.class_mapping.keys()) to ensure proper index mapping.
                c_index = list(self.class_mapping.keys()).index(c_i)
                num_left[c_index] += 1
                num_right[c_index] -= 1
                # Compute Gini impurity for left and right partitions.
                gini_left = 1.0 - sum((num_left[x] / i) ** 2 for x in range(len(num_parent)))
                gini_right = 1.0 - sum((num_right[x] / (m - i)) ** 2 for x in range(len(num_parent)))
                # Calculate the weighted Gini impurity.
                gini = (i * gini_left + (m - i) * gini_right) / m

                # Skip if threshold value is identical to the previous one.
                if threshold[i] == threshold[i - 1]:
                    continue

                # Update best split if a lower impurity is found.
                if gini < best_gini and ((best_gini - gini) >= self.min_impurity_decrease):
                    best_gini = gini
                    best_idx = idx
                    best_t = (threshold[i] + threshold[i - 1]) / 2

        return best_idx, best_t

    # Recursively build the decision tree.
    def _growtree(self, X, y, depth=0):
        num_samples_per_class = [np.sum(y == c) for c in self.class_mapping.keys()]
        # Majority vote for prediction.
        predicted = np.argmax(num_samples_per_class)
        node = {"Predicted": list(self.class_mapping.keys())[predicted]}

        # Check if the maximum depth is not reached.
        if depth < self.max_depth:
            idx, t = self._best_split(X, y)
            if idx is not None:
                (X_left, y_left), (X_right, y_right) = self._split(X, y, idx, t)
                # Check if the node meets the minimum samples requirement.
                if len(y_left) >= self.min_samples_split and len(y_right) >= self.min_samples_split:
                    node["feature_index"] = idx
                    node["threshold"] = t
                    # Recursively create left and right subtrees.
                    node["left"] = self._growtree(X_left, y_left, depth + 1)
                    node["right"] = self._growtree(X_right, y_right, depth + 1)
                    # Update feature importance.
                    impurity_decrease = self._gini(y) - (
                        len(y_left) / len(y) * self._gini(y_left) +
                        len(y_right) / len(y) * self._gini(y_right)
                    )
                    self.feature_importances_[idx] += impurity_decrease
        return node

    # Recursively predict the class label for a given sample x.
    def _predict(self, x, tree):
        if "threshold" in tree:
            if x[tree["feature_index"]] <= tree["threshold"]:
                return self._predict(x, tree["left"])
            else:
                return self._predict(x, tree["right"])
        else:
            return tree["Predicted"]

    # Calculate the accuracy of predictions.
    def accuracy(self, y_pred, y):
        return sum(y_pred == y) / y.shape[0]

    # Return normalized feature importances.
    def feature_importances(self):
        total_importance = np.sum(self.feature_importances_)
        return (
            self.feature_importances_ / total_importance
            if total_importance != 0
            else np.zeros_like(self.feature_importances_)
        )

# RandomForest class implements an ensemble of decision trees.
class RandomForest:
    def __init__(self, n_estimators=10, max_features="sqrt", max_depth=5, min_samples_split=2, min_impurity_decrease=0.0, random_state=None):
        # Initialize the random forest parameters.
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_impurity_decrease = min_impurity_decrease
        self.trees = []
        self.random_state = random_state
        if self.random_state is not None:
            np.random.seed(random_state)

    # Build a single decision tree using bootstrap sampling and random feature selection.
    def _build_tree(self, X, y):
        n_samples, n_features = X.shape
        selected_features = self._max_features(n_features)
        # Generate a bootstrap sample of the indices.
        indices = np.random.choice(n_samples, n_samples, replace=True)
        # Randomly select a subset of features.
        features = np.random.choice(n_features, selected_features, replace=False)
        X_sample = X[indices][:, features]
        y_sample = y[indices]
        tree = DecisionTree(
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_impurity_decrease=self.min_impurity_decrease
        )
        tree.fit(X_sample, y_sample)
        # Store the features used by this tree.
        tree.features = features
        return tree

    # Fit the random forest model by building multiple trees.
    def fit(self, X, y):
        self.trees = [self._build_tree(X, y) for _ in range(self.n_estimators)]

    # Determine the number of features to use based on the max_features parameter.
    def _max_features(self, n_features):
        if isinstance(self.max_features, int):
            return self.max_features
        elif self.max_features == "sqrt":
            return int(np.sqrt(n_features))
        elif self.max_features == "log2":
            return int(np.log2(n_features))
        else:
            return n_features

    # Predict labels from a single tree.
    def _predict_tree(self, tree, X):
        return [tree._predict(inputs, tree.tree) for inputs in X]

    # Predict class labels by aggregating predictions from all trees using majority vote.
    def predict(self, X):
        predicted = [self._predict_tree(tree, X) for tree in self.trees]
        return np.array([np.bincount(i).argmax() for i in zip(*predicted)])