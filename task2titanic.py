import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load the Titanic dataset from a CSV file
titanic_data = pd.read_csv(r'titanic.csv')

# Display the first few rows of the dataset to understand its structure
print(titanic_data.head())

# Drop irrelevant columns for simplicity (you may want to explore and preprocess more)
titanic_data = titanic_data.drop(['Name', 'Ticket', 'Cabin', 'Embarked'], axis=1)

# Convert categorical features to numerical values (e.g., Sex)
titanic_data['Sex'] = titanic_data['Sex'].map({'female': 0, 'male': 1})

# Handle missing values (you may want to explore more sophisticated strategies)
titanic_data = titanic_data.fillna(method='ffill')

# Split the dataset into features (X) and target variable (y)
X = titanic_data.drop('Survived', axis=1)
y = titanic_data['Survived']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a RandomForestClassifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the classifier
clf.fit(X_train, y_train)

# Make predictions on the test set
y_pred = clf.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

# Display the results
print(f"Accuracy: {accuracy:.2f}")
print("\nClassification Report:")
print(report)
