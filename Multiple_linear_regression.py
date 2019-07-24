import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def feature_scaling(value):
    mean = X.mean(axis=0)
    stdev = X.std(axis=0)
    X = (X - mean)/stdev
    return X

def gradient_descent(theta, X, y):
    X = feature_scaling(X)

def main():
    data = pd.read_csv("data.csv")
    X = data.values[:, :2]
    y = np.array(data.price)
    theta = np.zeros(3)
    X = np.c_[np.ones(X.shape[0]), X]
    theta = gradient_descent(theta, X, y)

if __name__ == "__main__":
	main();

