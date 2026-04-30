# we are creating simple and easy ml model:
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
# step 2 creating dataset
np.random.seed(0)
height = np.random.normal(160, 10, 100)
weight = 0.6 * height + np.random.normal(0, 5, 100)
# step 3 splitting dataset for training and testing
X = height.reshape(-1, 1)
y = weight
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
# step 4 creating and training the model
model = LinearRegression()
model.fit(X_train, y_train) 
# step 5 making predictions
y_pred = model.predict(X_test)
# step 6 visualizing the results
plt.scatter(X_test, y_test, color='blue', label='Actual Data')
plt.plot(X_test, y_pred, color='red', label='regression line')
plt.xlabel('Height')
plt.ylabel('Weight')
plt.title('Linear Regression: Height vs Weight')
plt.legend()
plt.show()