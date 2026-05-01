import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder

#step2 data set:
hours_Studied = [2.5, 1.5, 3.0, 1.8, 4.0, 2.0, 3.5, 2.7]
previous_exam = [80, 70, 7.5, 60, 85, 80, 90, 65]
exam_outcome = ['pass', 'fail', 'pass', 'fail', 'pass', 'pass', 'pass', 'fail']

label_encoder = LabelEncoder()
encoded_exam_outcome = label_encoder.fit_transform(exam_outcome)
X = np.column_stack((hours_Studied, previous_exam))
y = encoded_exam_outcome
clf = MLPClassifier(hidden_layer_sizes=(4,), activation='logistic', max_iter=1000, random_state=42)
clf.fit(X, y)

new_Student_data = np.array([[2.0, 78]]) #hours studied previous exam score
predicted_outcome = clf.predict(new_Student_data)

predicted_outcome_decode = label_encoder.inverse_transform(predicted_outcome)
print("predicted exam outcome for the new student is : {}".format(predicted_outcome_decode[0]))