from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
# Sample dataset
emails = [
    "Buy Cheap watches now",
    "hi can we reschedule our meeting?",
    "congratulations you won a lottery",
    "please find attached report for your review",
    "limited time offer, buy now!",
    "let's catch up tomorrow"
]

labels = ["spam", "not spam", "spam", "not spam", "spam", "not spam"]
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(emails)
X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.3, random_state=42)

naive_bayes_model = MultinomialNB()
naive_bayes_model.fit(X_train, y_train)

#predection:
y_pred = naive_bayes_model.predict(X_test)
#model evaluation:
accuracy = accuracy_score(y_test, y_pred)
confusion = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred)
print("Accuracy:", accuracy)
print("Confusion Matrix:\n", confusion)
print(confusion)
print("Classification Report:\n", report)
print(report)