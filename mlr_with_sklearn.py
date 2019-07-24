# Regression Linéaire Multiple

# Importer les librairies
import numpy as np
import pandas as pd

# Importer le dataset
dataset = pd.read_csv('data.csv')
X = dataset.iloc[:, :2].values
y = dataset.iloc[:, -1].values
# Diviser le dataset entre le Training set et le Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Construction du modèle
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Faire de nouvelles prédictions
y_pred = regressor.predict(X_test)
regressor.predict(np.array([[1650, 3]]))
