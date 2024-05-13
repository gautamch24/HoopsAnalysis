import pandas as pd # For data manipulation and analysis.
from sklearn.model_selection import train_test_split # To split the data into training and testing sets.
from sklearn.preprocessing import MinMaxScaler # To normalize the feature data.
from sklearn.linear_model import RidgeClassifier # A classifier that uses Ridge regression.
from sklearn.metrics import accuracy_score # To evaluate the accuracy of the model.
from sklearn.pipeline import Pipeline # To sequentially apply a list of transforms and a final estimator.
import joblib # For saving and loading the model.

# Load your dataset
data_path = 'pastNbaGames.csv'  # path to your dataset
data = pd.read_csv(data_path)  # read data set into panda data frame
# Loads the data from CSV file into a DataFrame.

# Select features and target
features = [
    'Pace', 'eFG%', 'TOV%', 'ORB%', 'FT/FGA', 'ORtg', 'Home',
    'Pace_opp', 'eFG%_opp', 'TOV%_opp', 'ORB%_opp', 'FT/FGA_opp', 'ORtg_opp', 'Home_opp'
]
target = 'won'
# Specifies which columns in the dataset are used as features for training and which column is the target (outcome).

# Fill missing values if any
# This line ensures that we only calculate the mean for numeric columns
numeric_data = data.select_dtypes(include='number')
data.fillna(numeric_data.mean(), inplace=True)
# Fills missing values with the mean of each column to ensure the model has complete data for training.

# Convert boolean 'won' into integers
data[target] = data[target].astype(int)
# Ensures the target variable is in integer format, which is necessary for the classifier to function correctly.

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(data[features], data[target], test_size=0.3, random_state=42)
# Divides the data into training and testing sets, with 30% of the data reserved for testing the model.

# Create a pipeline with scaler and model
model_pipeline = Pipeline([
    ('scaler', MinMaxScaler()),
    ('classifier', RidgeClassifier())
])
# Creating a pipeline that first scales the data then applies the Ridge classifier.

# Train the pipeline on training data
model_pipeline.fit(X_train, y_train)

# Predict on the test set
predictions = model_pipeline.predict(X_test)
print("Accuracy:", accuracy_score(y_test, predictions))
# Using the trained pipeline to make predictions on the test data.

# Save the trained model to a file
model_path = 'ridge_classifier_model.pkl'
joblib.dump(model_pipeline, model_path)
print("Model saved to", model_path)
# Saves the trained model to a file using joblib for later use in making predictions on new data.

