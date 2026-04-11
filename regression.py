import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

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

# Plot outputs
# This plots your actual, true data points on the graph. The training data (what the model learned from) will be black dots, 
# and the test data (what we are evaluating on) will be red dots.
plt.scatter(X_train, y_train, color='black', label='Train data points')
plt.scatter(X_test, y_test, color='red', label='Test data points')

# This draws the actual line of best fit (the y = mx + b equation) that your model calculated as a solid blue line.
# And plots the model's guesses for the test data as red "x" marks.
plt.plot(X_test, y_pred_test, color='blue', linewidth=1, label='Model')
plt.scatter(X_test, y_pred_test, marker='x', color='red', linewidth=3, label='Test Pred.')
plt.legend()
#plt.show()


# Looking at the difference between the red x's (the model's predictions) 
# and the red dots (the actual testing datapoints), you can visually confirm that the predictions are significantly off. 
# This visual gap is exactly what the Mean Squared Error (MSE) measures, and it is why the r^2 score is negative. 
# A good result would have the red x's sitting very close to the red dots, resulting in a positive r^2 close to +1


# Model Normalization: a crucial step in ML
# Normalizing your data, meaning, scaling it down so all the numbers are on a similar playing field, is highly desirable, 
# especially when you want to interpret the difference in effect between various model features. Use numpy for this.
mean_X = np.mean(X_train, axis=0)
std_X = np.std(X_train, axis=0)
mean_y = np.mean(y)
std_y = np.std(y)

# StandardScaler() packages all the math for mean, std into a single object that "remembers" the exact scaling parameters of 
# your training data so you can perfectly apply them to your testing data.
# Intialize scaler
scaler = StandardScaler()

# "Fit" the scaler to the training data (learn the mean/std) AND transform the data (scale it) in one step
# Used only on training data
X_train_norm = scaler.fit_transform(X_train)

# Train the model on the normalized data
regr = LinearRegression(fit_intercept=True)
regr.fit(X_train_norm, y_train)

# Transform the test data using the parameters learned in step 2
# Used on testing data. Strictly applies the math it learned from the training set.
X_test_norm = scaler.transform(X_test)

# Predict and evaluate
y_pred_test = regr.predict(X_test_norm)
print('Clean Pipeline MSE: %.2f' % mean_squared_error(y_test, y_pred_test))

# Now that the model is trained on normalized data, it expects any new data to be normalized as well. 
# The tutorial demonstrates what happens if you forget this rule by predicting directly on X_test
y_pred_bad = regr.predict(X_test)
print('Bad MSE: %.2f' % mean_squared_error(y_test, y_pred_bad))
# The model is completely confused because it learned on small, scaled numbers but is being tested on large, raw numbers.

# THE FIX
# 1. Normalize the test input (just like we did the training data!)
X_test_norm = (X_test - mean_X) / std_X

# 2. Predict using the normalized test data
y_pred_test_norm = regr.predict(X_test_norm)

# 3. Unnormalize y after prediction!
y_pred_test = (y_pred_test_norm * std_y) + mean_y

print('Correct MSE: %.2f' % mean_squared_error(y_test, y_pred_test))

plt.scatter(X_test, y_test, label='True')
plt.scatter(X_test, y_pred_test, label='Predicted')
plt.legend()
plt.show()