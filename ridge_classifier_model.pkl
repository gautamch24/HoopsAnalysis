import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import accuracy_score
import joblib

# pandas: For data manipulation and analysis.
# train_test_split: To split the data into training and testing sets.
# MinMaxScaler: To normalize the feature data.
# RidgeClassifier: A classifier that uses Ridge regression.
# accuracy_score: To evaluate the accuracy of the model.
# joblib: For saving and loading the model.


# Load your dataset
data_path = 'pastNbaGames.csv'  #  path to your dataset
data = pd.read_csv(data_path)
# Loads the data from CSV file into a DataFrame.


# Select features and target
features = [
    'Pace', 'eFG%', 'TOV%', 'ORB%', 'FT/FGA', 'ORtg', 'Home',
    'Pace_opp', 'eFG%_opp', 'TOV%_opp', 'ORB%_opp', 'FT/FGA_opp', 'ORtg_opp', 'Home_opp'
]
target = 'won'  # Update if target variable name is different
# Specifies which columns in the dataset are used as features for training and which column is the target (outcome).


# Fill missing values if any
data.fillna(data.mean(), inplace=True)
# Fills missing values with the mean of each column to ensure the model has complete data for training.


# Convert boolean 'won' into integers 
data[target] = data[target].astype(int)
# Ensures the target variable is in integer format, which is necessary for the classifier to function correctly.


# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(data[features], data[target], test_size=0.3, random_state=42)
# Divides the data into training and testing sets, with 30% of the data reserved for testing the model.


# Scale features
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
# Normalizes the feature data to ensure that all features contribute equally to the model training, improving performance.


# Initialize and train the Ridge Classifier
ridge_classifier = RidgeClassifier()
ridge_classifier.fit(X_train_scaled, y_train)
# Creates a Ridge Classifier and trains it on the normalized training data.


# Predict on the test set
predictions = ridge_classifier.predict(X_test_scaled)
print("Accuracy:", accuracy_score(y_test, predictions))
# Makes predictions on the testing set and prints the accuracy of the model.


# Save the trained model to a file
model_path = 'ridge_classifier_model.pkl'
joblib.dump(ridge_classifier, model_path)
print("Model saved to", model_path)
# Saves the trained model to a file using joblib for later use in making predictions on new data.
