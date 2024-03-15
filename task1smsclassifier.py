import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.pipeline import Pipeline

# Download NLTK resources
import nltk
nltk.download('stopwords')

# Load the SMS Spam Collection dataset
sms_data = pd.read_csv('spam.csv', encoding='latin-1')

# Display the first few rows of the dataset to understand its structure
print(sms_data.head())

# Drop irrelevant columns and rename the remaining columns
sms_data = sms_data[['v1', 'v2']]
sms_data.columns = ['label', 'text']

# Convert labels to binary values (0 for 'ham', 1 for 'spam')
sms_data['label'] = sms_data['label'].map({'ham': 0, 'spam': 1})

# Split the dataset into features (X) and target variable (y)
X = sms_data['text']
y = sms_data['label']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a text classification pipeline with a naive Bayes classifier
text_clf = Pipeline([
    ('vectorizer', CountVectorizer(analyzer='word', stop_words=stopwords.words('english'), max_features=5000)),
    ('classifier', MultinomialNB())
])

# Train the classifier
text_clf.fit(X_train, y_train)

# Make predictions on the test set
y_pred = text_clf.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

# Display the results
print(f"Accuracy: {accuracy:.2f}")
print("\nClassification Report:")
print(report)
