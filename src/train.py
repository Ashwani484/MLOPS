import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error, r2_score
import mlflow
import mlflow.sklearn
import psutil as ps
import warnings

# Suppress potential convergence warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning)


# --- 1. Load the data correctly ---
print("--- Loading Iris dataset using scikit-learn ---")
iris_bunch = load_iris()
X = iris_bunch.data
y = iris_bunch.target


# --- 2. Split and scale the data ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# --- 3. Define models ---
models = {
    "Logistic Regression": LogisticRegression(max_iter=200),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier()
}

# Set the MLflow experiment
mlflow.set_experiment("Iris-Classification-1.0")


# --- 4. Initialize variables to track the best model ---
best_accuracy = 0.9
best_model_run_id = None
best_model_artifact_path = None


# --- 5. Train models and log with MLflow ---
for model_name, model in models.items():
    print(f"\n--- Training {model_name} ---")
    
    with mlflow.start_run(run_name=model_name) as run:
        # Fit the model and make predictions
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        
        # Calculate and log metrics
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Accuracy: {accuracy:.4f}")
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("mse", mean_squared_error(y_test, y_pred))
        mlflow.log_metric("r2_score", r2_score(y_test, y_pred))
        
        # Log parameters and system info
        mlflow.log_params(model.get_params())
        mlflow.log_param("cpu_usage_at_start", ps.cpu_percent())
        mlflow.log_param("ram_usage_at_start", ps.virtual_memory().percent)
        
        # Log the model
        artifact_path = "model" # A generic artifact path
        mlflow.sklearn.log_model(model, artifact_path=artifact_path)
        
        # --- Check if this is the best model so far ---
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model_name=model_name
            best_model_run_id = run.info.run_id
            best_model_artifact_path = artifact_path
            print(f"New best model found: {model_name} with accuracy: {best_accuracy:.4f}")

print("\n--- Training and logging complete. ---")


# --- 6. Register the best model in the MLflow Model Registry ---
if best_model_run_id:
    registered_model_name = str("iris-best-classifier"+best_model_name)
    model_uri = f"runs:/{best_model_run_id}/{best_model_artifact_path}"
    
    print(f"\nRegistering the best model '{registered_model_name}' from run '{best_model_run_id}'.")
    print(f"Model URI: {model_uri}")
    
    try:
        # Register the model
        mlflow.register_model(model_uri=model_uri, name=registered_model_name)
        print(f"\nModel '{registered_model_name}' registered successfully!")
    except Exception as e:
        print(f"An error occurred while registering the model: {e}")
else:
    print("No model was trained successfully to register.")

