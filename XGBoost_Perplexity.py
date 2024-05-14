import pandas as pd
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.metrics import accuracy_score
from sklearn.compose import ColumnTransformer

# Data preprocessing function
def preprocess_data(file_path):
    # Load data
    data = pd.read_csv(file_path)

    # Handle missing values
    data = data.dropna()

    # Drop the date column
    if 'date' in data.columns:
        data = data.drop(columns=['date'])

    # Encode categorical features
    categorical_cols = ['team', 'team_opp']
    numerical_cols = [col for col in data.columns if col not in categorical_cols + ['won']]

    # Label encoding for target variable
    label_encoder = LabelEncoder()
    data['won'] = label_encoder.fit_transform(data['won'])

    # One-hot encoding for categorical features
    categorical_transformer = OneHotEncoder(handle_unknown='ignore')
    
    # Create column transformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', categorical_transformer, categorical_cols),
            ('num', 'passthrough', numerical_cols)
        ])

    # Transform data
    X = preprocessor.fit_transform(data)
    y = data['won']

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test, preprocessor

# Model training function
def train_model(X_train, X_test, y_train, y_test):
    # Create DMatrix objects
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)

    # Set XGBoost parameters
    params = {
        'objective': 'binary:logistic',  # Binary classification
        'max_depth': 6,  # Maximum depth of a tree
        'eta': 0.3,  # Learning rate
        'subsample': 0.8,  # Subsample ratio of columns for each tree
        'colsample_bytree': 0.8,  # Subsample ratio of columns for each level
        'eval_metric': 'auc'  # Evaluation metric
    }

    # Train the model
    num_rounds = 100  # Number of boosting rounds
    model = xgb.train(params, dtrain, num_rounds, evals=[(dtest, 'test')], early_stopping_rounds=10, verbose_eval=10)

    return model

# Main function
def main():
    # Load data and preprocess
    X_train, X_test, y_train, y_test, preprocessor = preprocess_data("pastNbaGames.csv")

    # Train the model
    model = train_model(X_train, X_test, y_train, y_test)

    # Make predictions on test data
    y_pred = model.predict(xgb.DMatrix(X_test))
    predictions = [round(value) for value in y_pred]

    # Evaluate model performance
    accuracy = accuracy_score(y_test, predictions)
    print(f"Accuracy: {accuracy * 100:.2f}%")

    # Make predictions on new data
    # Note: Replace the placeholder with actual new data loading logic
    new_data = pd.DataFrame({
        'team': ['team1'],
        'team_opp': ['team2'],
        # Add all the numerical features here with appropriate values
        # 'feature1': [value1],
        # 'feature2': [value2],
        # ...
    })
    
    new_data_transformed = preprocessor.transform(new_data)
    new_predictions = model.predict(xgb.DMatrix(new_data_transformed))

    # Print predictions
    print("Predictions:")
    for pred in new_predictions:
        print(f"Predicted outcome: {'Win' if pred > 0.5 else 'Loss'}")

if __name__ == "__main__":
    main()
