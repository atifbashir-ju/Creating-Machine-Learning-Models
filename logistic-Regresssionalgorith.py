# Create a logistic regression model:
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

#step 2 preparing dataset
hours_studied = np.array([2, 3, 5, 1, 8, 10 ,7, 6, 4, 9])
pass_fail = np.array([0, 0, 1, 0, 1, 1, 1, 1, 0, 1])
# step 3 splitting dataset for training and testing
X_train, X_test, y_train, y_test = train_test_split(hours_studied.reshape(-1, 1), pass_fail, test_size=0.2, random_state=42)
# step 4 creating and training the model
model = LogisticRegression()
model.fit(X_train, y_train)
# step 5 making predictions
predictions = model.predict(X_test)
# step 6 evaluating the model
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, predictions)
print('Accuracy:', accuracy)