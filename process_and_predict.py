import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import RidgeClassifier
import joblib

# Define a function to preprocess and predict using new data
def process_and_predict(data_path, model_path, output_path):
# Defines a function named process_and_predict that takes paths to the data, the model, and where to save the output.


    # Load new data
    data = pd.read_csv(data_path)


    # Define the features expected by the model
    features = [
        'Pace', 'eFG%', 'TOV%', 'ORB%', 'FT/FGA', 'ORtg', 'Home',
        'Pace_opp', 'eFG%_opp', 'TOV%_opp', 'ORB%_opp', 'FT/FGA_opp', 'ORtg_opp', 'Home_opp'
    ]

    # Fill missing values with the mean 
    data[features] = data[features].fillna(data[features].mean())


    # IF NO PIPELINE 
    # Scale the features using MinMaxScaler (assuming this was used during model training)
    # scaler = MinMaxScaler()
    # data[features] = scaler.fit_transform(data[features])
# Defines the features used by the model, fills missing values, and scales the features similarly to the training process.
    # Load the trained Ridge Classifier model
    # model = joblib.load(model_path)
    # IF NI PIPELINE


    # Load the trained pipeline (model + scaler)
    model_pipeline = joblib.load(model_path)
    
    # Predict the outcome
    data['predicted_won'] = model_pipeline.predict(data[features])
# Loads the saved model and uses it to make predictions on the new data.


    # Save the predictions to a new CSV file
    data.to_csv(output_path, index=False)
    print("Predictions saved to:", output_path)

# Usage example
if __name__ == "__main__":
    data_path = 'pastNbaGames.csv'  # data file path
    model_path = 'ridge_classifier_model.pkl'   # Path to saved model
    output_path = 'predictions.csv'             # Path to save the predictions

    process_and_predict(data_path, model_path, output_path)
# Saves the new data with predictions appended to a CSV file and prints the location of the saved file.
