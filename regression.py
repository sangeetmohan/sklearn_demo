import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Import some real data
X, y = datasets.load_diabetes(return_X_y=True)

# For a small example, only take 20 datapoints and a single feature column
# grab first 20 rows and one explanatory variable (at index 2)
X = X[:20, [2]]
y = y[:20]

# Split the dataset into training and testing sets
# train_test_split is one of the most used functions in ML. 
# it splits dataset into 2 buckets: 
# 1. one that is used to train the model
# 2. another that is used to test how well the model generalizes
# Setting random_state=0 ensures you get the exact same "random" split every time you run the script, 
# which is perfect for reproducible results
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

# Print the shapes to verify the split
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
print(X_train[:5]) # Show first five data points

# Fit linear regression model to data
# saying that we want to use lin reg model and setting fit_intercept = True says we don't 
# need line of best fit to cross the origin exactly
# fit_intercept=True is important because it lets the model naturally find the y intercept
regr = LinearRegression(fit_intercept=True)
# .fit is where the training happens
# scikit does calculus and lin alg behind the scenes to calculate the line of best fit
regr.fit(X_train, y_train)

# Inspect the parameters of the trained model
# .coef_ is the coefficient 'm' for y = mx + b
# .intercept is the 'b' for y = mx + b
print('Coefficients:', regr.coef_)
print('Intercept:', regr.intercept_)

# Predict y for test (new) data points
# regr.predict(X_test) gives the trained model unseen data (X_test)
# it predicts the y-value for these x values by plugging into y = mx + b
y_pred_test = regr.predict(X_test)
print("Predictions shape:", y_pred_test.shape)
print("Predictions:", y_pred_test)

# Evaluating the model: MSE and r²
# MSE calculates the distance between the predicted point and actual point. It squares those distances to remove negative
# signs and heavily penalize huge errors, and finds the average. A lower MSE is better.

# r^2 tells how much of the variance in the target variable is explained by your feature. 
# A perfect prediction would have an r^2 of 1.0.
print('Mean squared error: %.2f' % mean_squared_error(y_test, y_pred_test))
print('Coefficient of determination: %.2f' % r2_score(y_test, y_pred_test))