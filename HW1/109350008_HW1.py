import numpy as np
import matplotlib.pyplot as plt
from pandas import DataFrame, read_csv

class LinearRegression:
    def __init__(self):
        self.closed_form_weights = None
        self.closed_form_intercept = None
        self.gradient_descent_weights = None
        self.gradient_descent_intercept = None
        self.epoch_history = []
        self.GD_loss_history = []
        
    # This function computes the closed-form solution of linear regression.
    def closed_form_fit(self, X, y):
        # Compute closed-form solution.
        # Save the weights and intercept to self.closed_form_weights and self.closed_form_intercept
        sample, feature = X.shape
        X = np.hstack((np.ones((sample, 1)), X))

        self.closed_form_weights = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y) # all weight: w0, w1, w2....(include intercept w0)
        
        self.closed_form_intercept = self.closed_form_weights[0]
        self.closed_form_weights = self.closed_form_weights[1::]

    # This function computes the gradient descent solution of linear regression.
    def gradient_descent_fit(self, X, y, lr, epochs):
        # Initialize weights and intercept
        sample, feature = X.shape
        X = np.hstack((np.ones((sample, 1)), X))

        self.gradient_descent_weights = np.zeros(feature + 1)
        
        for epoch in range(epochs):
            # Compute predictions
            predictions = X.dot(self.gradient_descent_weights.T)

            # loss
            loss = self.get_mse_loss(predictions, y)
            self.GD_loss_history.append(loss)

            # Compute gradients
            #print("loss: ", loss) ##
            #print("error: ", predictions - y) ##
            gradient_weights = 2/sample * np.sum((predictions - y) * X.T, axis = 1)
            #print(gradient_weights) ##
            
            # Update
            self.gradient_descent_weights -= lr * gradient_weights.T
            self.epoch_history.append(epoch)
        
        #print("weight_after: ", self.gradient_descent_weights) ##
        self.gradient_descent_intercept = self.gradient_descent_weights[0]
        self.gradient_descent_weights = self.gradient_descent_weights[1::]
    
    # This function compute the MSE loss value between your prediction and ground truth.
    def get_mse_loss(self, prediction, ground_truth):
        return np.mean((prediction - ground_truth)**2)

    # This function takes the input data X and predicts the y values according to your closed-form solution.
    def closed_form_predict(self, X):
        return np.dot(X, self.closed_form_weights) + self.closed_form_intercept

    # This function takes the input data X and predicts the y values according to your gradient descent solution.
    def gradient_descent_predict(self, X):
        return np.dot(X, self.gradient_descent_weights) + self.gradient_descent_intercept
    
    # This function takes the input data X and predicts the y values according to your closed-form solution, 
    # and returns the MSE loss between the prediction and the input y values.
    def closed_form_evaluate(self, X, y):
        return self.get_mse_loss(self.closed_form_predict(X), y)

    # This function takes the input data X and predicts the y values according to your gradient descent solution, 
    # and returns the MSE loss between the prediction and the input y values.
    def gradient_descent_evaluate(self, X, y):
        return self.get_mse_loss(self.gradient_descent_predict(X), y)
        
    # This function uses matplotlib to plot and show the learning curve (x-axis: epoch, y-axis: training loss) of your gradient descent solution.
    def plot_learning_curve(self):

        plt.plot(range(len(self.epoch_history)), self.GD_loss_history)
        plt.xlabel('Epoch')
        plt.ylabel('Training Loss')
        plt.title('Learning Curve')
        plt.show()

# Do not modify the main function architecture.
# You can only modify the arguments of your gradient descent fitting function.
if __name__ == "__main__":
    # Data Preparation
    train_df = DataFrame(read_csv("train.csv"))
    train_x = train_df.drop(["Performance Index"], axis=1)
    train_y = train_df["Performance Index"]
    train_x = train_x.to_numpy()
    train_y = train_y.to_numpy()
    
    # Model Training and Evaluation
    LR = LinearRegression()

    LR.closed_form_fit(train_x, train_y)
    print("Closed-form Solution")
    print(f"Weights: {LR.closed_form_weights}, Intercept: {LR.closed_form_intercept}")

    LR.gradient_descent_fit(train_x, train_y, lr=1e-4, epochs=800000)
    print("Gradient Descent Solution")
    print(f"Weights: {LR.gradient_descent_weights}, Intercept: {LR.gradient_descent_intercept}")

    test_df = DataFrame(read_csv("test.csv"))
    test_x = test_df.drop(["Performance Index"], axis=1)
    test_y = test_df["Performance Index"]
    test_x = test_x.to_numpy()
    test_y = test_y.to_numpy()

    closed_form_loss = LR.closed_form_evaluate(test_x, test_y)
    gradient_descent_loss = LR.gradient_descent_evaluate(test_x, test_y)
    #print("close mse: ", closed_form_loss) ##
    #print("gradient mse: ", gradient_descent_loss) ##
    print(f"Error Rate: {((gradient_descent_loss - closed_form_loss) / closed_form_loss * 100):.1f}%")

    # Plot the learning curve
    LR.plot_learning_curve()