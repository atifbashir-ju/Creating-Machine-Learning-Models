import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Sample dataset: Each row represents (feature1, feature2) and the corresponding label (0 or 1)
X = np.array([[100, 1], [50, 0], [75, 1], [200, 1], [30, 0],
              [180, 1], [120, 0], [60, 1], [110, 1], [90, 0], [150, 0]])

y = np.array([1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 0])

# split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# model creation
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)

# train the model
rf_classifier.fit(X_train, y_train)

# predict the labels for the test set
y_pred = rf_classifier.predict(X_test)

# accuracy evaluation
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy*100:.2f}%")