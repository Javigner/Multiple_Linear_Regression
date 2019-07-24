import pandas as pd
import numpy as np

def standar_scaler(X):
    mean = X.mean(axis=0)
    stdev = X.std(axis=0)
    X = (X - mean)/stdev
    return X, mean, stdev

def cost_function(theta0, theta1, theta2, X, y, key):
    result = 0
    for i in range(len(X)):
        result += ((theta0 * X[i][0] + theta1 * X[i][1] + theta2 * X[i][2] - y[i]) * X[i][key])
    return result

def prediction(theta, pred):
    predict = theta[0] * pred[0] + theta[1] * pred[1] + theta[2] * pred[2]
    return predict

def gradient_descent(theta, X, y):
    X, mean, stdev = standar_scaler(X)
    X = np.c_[np.ones(X.shape[0]), X]
    testha = [1.0, 1.0, 1.0]
    learning_rate = 0.1
    while (testha[0] != theta[0] and testha[1] != theta[1] and testha[2] != theta[2]):
        theta[0] = testha[0]
        theta[1] = testha[1]
        theta[2] = testha[2]
        testha[0] = theta[0] - learning_rate * (1.0 / len(X)) * cost_function(theta[0], theta[1], theta[2], X, y, 0)
        testha[1] = theta[1] - learning_rate * (1.0 / len(X)) * cost_function(theta[0], theta[1], theta[2], X, y, 1)
        testha[2] = theta[2] - learning_rate * (1.0 / len(X)) * cost_function(theta[0], theta[1], theta[2], X, y, 2)
    return theta, mean, stdev

def main():
    data = pd.read_csv("data.csv")
    X = data.values[:, :2]
    y = np.array(data.price)
    theta = np.zeros(3)
    theta, mean, stdev = gradient_descent(theta, X, y)
    X_test = (np.array([1650,3]) - mean) / stdev
    X_test = np.hstack([1, X_test])
    predict = prediction(theta, X_test) 
    print(predict)
        
if __name__ == "__main__":
	main();

