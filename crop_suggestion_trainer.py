import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
import os
import json

DATASET_CSV_PATH = 'Crop_recommendation.csv'
MODEL_SAVE_PATH = 'crop_recommendation_model.joblib'
FEATURES_SAVE_PATH = 'crop_features.json'

if not os.path.exists(DATASET_CSV_PATH):
    print(f"Error: Dataset CSV file not found at '{DATASET_CSV_PATH}'")
    print("Please make sure the file is in the same directory as this script.")
    exit()

print("Loading dataset...")
try:
    df = pd.read_csv(DATASET_CSV_PATH)
except Exception as e:
    print(f"Error reading CSV file: {e}")
    exit()

print("Dataset loaded successfully.")
print(f"Columns found: {df.columns.tolist()}")

required_features = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
target_column = 'label'

missing_features = [col for col in required_features if col not in df.columns]
if missing_features:
     print(f"Error: The dataset is missing required feature columns: {missing_features}")
     exit()

if target_column not in df.columns:
     print(f"Error: The dataset is missing the target column: '{target_column}'")
     exit()

print("Preparing data...")
X = df[required_features]
y = df[target_column]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Training data shape: {X_train.shape}")
print(f"Testing data shape: {X_test.shape}")
print(f"Unique crop labels found in dataset: {list(y.unique())}")


print("Training RandomForestClassifier model...")
model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

print("Evaluating model...")
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy on Test Set: {accuracy * 100:.2f}%")

print("Saving model...")
try:
    joblib.dump(model, MODEL_SAVE_PATH)
    print(f"Model saved successfully as '{MODEL_SAVE_PATH}'")
except Exception as e:
    print(f"Error saving model: {e}")
    exit()

print("Saving feature list...")
try:
    features_list = required_features
    with open(FEATURES_SAVE_PATH, 'w') as f:
        json.dump(features_list, f)
    print(f"Feature list saved successfully as '{FEATURES_SAVE_PATH}'")
except Exception as e:
    print(f"Error saving feature list: {e}")

print("\nTraining complete.")