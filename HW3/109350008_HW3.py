# You are not allowed to import any additional packages/libraries.
import numpy as np
from pandas import DataFrame, read_csv
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

# This function computes the gini impurity of a label array.
def gini(y, sample_weight=None):
    if sample_weight is not None:
        probabilities = np.bincount(y, weights=sample_weight) / np.sum(sample_weight)
    else:
        probabilities = np.bincount(y) / len(y)

    gini_index = 1 - np.sum(probabilities**2)
    return gini_index

def entropy(y, sample_weight=None):
    if sample_weight is not None:
        probabilities = np.bincount(y, weights=sample_weight) / np.sum(sample_weight)
    else:
        probabilities = np.bincount(y) / len(y)
    
    probabilities[probabilities == 0] = 1e-10

    entropy_value = -np.sum(probabilities * np.log2(probabilities))
    return entropy_value
        
# The decision tree classifier class.
# Tips: You may need another node class and build the decision tree recursively.
class DecisionTree():
    def __init__(self, criterion='gini', max_depth=None):
        self.criterion = criterion
        self.max_depth = max_depth 
        self.tree = None
    
    # This function computes the impurity based on the criterion.
    def impurity(self, y, sample_weight=None):
        if self.criterion == 'gini':
            return gini(y, sample_weight)
        elif self.criterion == 'entropy':
            return entropy(y, sample_weight)
    
    # This function fits the given data using the decision tree algorithm.
    def fit(self, X, y, depth = 0, sample_weight = None):
        if depth == self.max_depth or len(np.unique(y)) == 1:
            # If the maximum depth is reached or the node is pure, create leaf node.
            return {'class': np.argmax(np.bincount(y, weights=sample_weight)), 'leaf': True}

        best_feature, best_value = self.find_best_split(X, y, sample_weight)
        if best_feature is None:
            # If no split improves purity, create leaf node.
            return {'class': np.argmax(np.bincount(y, weights=sample_weight)), 'leaf': True}

        # Split the data by the best feature and value.
        left_indices = X[:, best_feature] <= best_value
        right_indices = ~left_indices

        # Recursively build subtrees.
        left_subtree = self.fit(X[left_indices], y[left_indices], depth + 1, sample_weight[left_indices] if sample_weight is not None else None)
        right_subtree = self.fit(X[right_indices], y[right_indices], depth + 1, sample_weight[right_indices] if sample_weight is not None else None)

        self.tree = {'feature': best_feature, 'value': best_value, 'left': left_subtree, 'right': right_subtree, 'leaf': False}
        return self.tree

    # This function finds the best split for the given data.
    def find_best_split(self, X, y, sample_weight=None):
        best_feature = None
        best_value = None
        best_impurity_reduction = 0
        
        for feature in range(X.shape[1]):
            unique_values = np.unique(X[:, feature])
            
            for value in unique_values:
                left_indices = X[:, feature] <= value
                right_indices = ~left_indices
                
                left_impurity = self.impurity(y[left_indices], sample_weight[left_indices] if sample_weight is not None else None)
                right_impurity = self.impurity(y[right_indices], sample_weight[right_indices] if sample_weight is not None else None)
                
                weighted_impurity = (len(y[left_indices]) * left_impurity + len(y[right_indices]) * right_impurity) / len(y)
                
                impurity_reduction = self.impurity(y, sample_weight) - weighted_impurity
                
                if impurity_reduction > best_impurity_reduction:
                    best_feature = feature
                    best_value = value
                    best_impurity_reduction = impurity_reduction
        
        return best_feature, best_value
    
    # This function takes the input data X and predicts the class label y according to your trained model.
    def predict(self, X):        
        predictions = np.zeros(X.shape[0])
        for i in range(X.shape[0]):
            current_node = self.tree
            while not current_node['leaf']:
                if X[i, current_node['feature']] <= current_node['value']:
                    current_node = current_node['left']
                else:
                    current_node = current_node['right']            
            predictions[i] = current_node['class']
        
        return predictions.astype(int)
        
    # This function plots the feature importance of the decision tree.
    def plot_feature_importance_img(self, columns):
        feature_counts = np.zeros(len(columns))
        self.count_feature_importance(self.tree, feature_counts)

        plt.barh(columns, feature_counts)
        plt.xlabel("Number of Splits")
        plt.ylabel("Feature")
        plt.title("Feature Importance")
        plt.show()

    def count_feature_importance(self, node, feature_counts):
        if node['leaf']:
            return
        feature_counts[node['feature']] += 1
        self.count_feature_importance(node['left'], feature_counts)
        self.count_feature_importance(node['right'], feature_counts)

# The AdaBoost classifier class.
class AdaBoost():
    def __init__(self, criterion='gini', n_estimators=200):
        self.criterion = criterion
        self.n_estimators = n_estimators
        self.alphas = []
        self.classifiers = []

    # This function fits the given data using the AdaBoost algorithm.
    # You need to create a decision tree classifier with max_depth = 1 in each iteration.
    def fit(self, X, y):
        # Initialize weights
        weights = np.ones(len(y)) / len(y)
        
        for _ in range(self.n_estimators):
            # Create a weak DT
            weak_classifier = DecisionTree(criterion=self.criterion, max_depth=1)
            weak_classifier.fit(X, y, sample_weight=weights)

            y_pred = weak_classifier.predict(X)
            
            # error
            error = np.sum(weights * (y_pred != y))
            
            # alpha
            alpha = 0.5 * np.log((1 - error) / error)

            # rescale y {0, 1} to {-1, 1}
            yt = y * 2 - 1
            yt_pred = y_pred * 2 - 1

            # Update weights
            weights *= np.exp(-alpha * yt * yt_pred)
            weights /= np.sum(weights)
            
            # Save the alpha and weak classifier
            self.alphas.append(alpha)
            self.classifiers.append(weak_classifier)

    # This function takes the input data X and predicts the class label y according to your trained model.
    def predict(self, X):
        # Initialize
        final_prediction = np.zeros(X.shape[0])

        for alpha, classifier in zip(self.alphas, self.classifiers):
            # Accumulate predictions
            final_prediction += alpha * classifier.predict(X)

        # normalize
        final_prediction /= np.sum(self.alphas)

        # Convert final prediction to binary
        final_prediction = np.where(final_prediction > 0.5, 1, 0)

        return final_prediction


# Do not modify the main function architecture.
# You can only modify the value of the random seed and the the arguments of your Adaboost class.
if __name__ == "__main__":
# Data Loading
    train_df = DataFrame(read_csv("train.csv"))
    test_df = DataFrame(read_csv("test.csv"))
    X_train = train_df.drop(["target"], axis=1)
    y_train = train_df["target"]
    X_test = test_df.drop(["target"], axis=1)
    y_test = test_df["target"]

    X_train = X_train.to_numpy()
    y_train = y_train.to_numpy()
    X_test = X_test.to_numpy()
    y_test = y_test.to_numpy()

# Set random seed to make sure you get the same result every time.
# You can change the random seed if you want to.
    np.random.seed(0)

# Decision Tree
    print("Part 1: Decision Tree")
    data = np.array([0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1])
    print(f"gini of [0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1]: {gini(data)}")
    print(f"entropy of [0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1]: {entropy(data)}")
    tree = DecisionTree(criterion='gini', max_depth=7)
    tree.fit(X_train, y_train)
    y_pred = tree.predict(X_test)
    print("Accuracy (gini with max_depth=7):", accuracy_score(y_test, y_pred))
    tree = DecisionTree(criterion='entropy', max_depth=7)
    tree.fit(X_train, y_train)
    y_pred = tree.predict(X_test)
    print("Accuracy (entropy with max_depth=7):", accuracy_score(y_test, y_pred))

    tree = DecisionTree(criterion='gini', max_depth=15)
    tree.fit(X_train, y_train)
    tree.plot_feature_importance_img(train_df.columns[:-1])

# AdaBoost
    print("Part 2: AdaBoost")
    # Tune the arguments of AdaBoost to achieve higher accuracy than your Decision Tree.
    ada = AdaBoost(criterion = 'entropy', n_estimators = 10)
    ada.fit(X_train, y_train)
    y_pred = ada.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
