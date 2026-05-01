#buildig a decision tree classifier model
from sklearn import tree
# each tuple represnets (outlook, temperature, humidity, playtennis)
data = [
    ("Sunny", "Hot", "High", "No"),
    ("Sunny", "Hot", "High", "No"),
    ("Overcast", "Hot", "High", "Yes"),
    ("Rainy", "Mild", "High", "Yes"),
    ("Rainy", "Cool", "Normal", "Yes"),
    ("Rainy", "Cool", "Normal", "No"),
    ("Overcast", "Cool", "Normal", "Yes"),
    ("Sunny", "Mild", "High", "No"),
    ("Sunny", "Cool", "Normal", "Yes"),
    ("Rainy", "Mild", "Normal", "Yes"),
    ("Sunny", "Mild", "Normal", "Yes"),
    ("Overcast", "Mild", "High", "Yes"),
    ("Overcast", "Hot", "Normal", "Yes"),
    ("Rainy", "Mild", "High", "No")
]
# converting categorical data to numerical data
outlook_mapping = {"Sunny": 0, "Overcast": 1, "Rainy": 2}
temperature_mapping = {"Hot": 0, "Mild": 1, "Cool": 2}
humidity_mapping = {"High": 0, "Normal": 1}
playtennis_mapping = {"No": 0, "Yes": 1}
#iterating over the original dataset
data_numeric = [(outlook_mapping[outlook], temperature_mapping[temperature], humidity_mapping[humidity], playtennis_mapping[playtennis]) 
                for outlook, temperature, humidity, playtennis in data]
# spilit data into features(X) and target variable(y)
X =[x[:-1] for x in data_numeric]
y = [x[-1] for x in data_numeric]

# now we will create out decision tree classifier model train the classifier on the data
clf = tree.DecisionTreeClassifier()
clf = clf.fit(X, y)
# predict wehather to play tennis or not based on the features
new_data_point = (outlook_mapping["Sunny"], temperature_mapping["Cool"], humidity_mapping["High"])
predicted = clf.predict([new_data_point])

#now map the predicted value back to the original label
predicted_label = [k for k, v in playtennis_mapping.items() if v == predicted[0]]
print("predicted label for the new data point:", predicted_label[0])