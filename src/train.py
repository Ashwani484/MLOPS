import pandas as pd
from logger import logging
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,LabelEncoder
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report,recall_score, precision_score,r2_score,mean_absolute_error,mean_squared_error
from sklearn.metrics import confusion_matrix
import mlflow
import mlflow.sklearn
import joblib
import os
import shutil
from get_data import get_data

logging.info("Starting the Loading Iris dataset...")
# --- (Data loading and model definitions are unchanged) ---

X,y=get_data()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
logging.info("train-test split done.")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
logging.info("Data scaling done.")
models = {
    "Logistic Regression": LogisticRegression(),
    "Random Forest": RandomForestClassifier(random_state=42)
    
}

logging.info("MLflow setup Started...")
# --- MLflow tracking for metrics only ---
mlflow.set_experiment("Iris-Classifier-Training")

best_accuracy = 0.0
best_model = None

# --- Train models, find the best one ---
for model_name, model in models.items():
    with mlflow.start_run(run_name=model_name):
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        mae=mean_squared_error(y_test,y_pred)
        r2=r2_score(y_test,y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1= f1_score(y_test, y_pred, average='weighted')
        confusion_matri = confusion_matrix(y_test, y_pred)
        print(f"Confusion Matrix for {model_name}:\n{confusion_matri}\n")
        print(f"Classification Report for {model_name}:\n{classification_report(y_test, y_pred)}\n")
        print(f"Precision for {model_name}: {precision:.4f}")
        print(f"Recall for {model_name}: {recall:.4f}")
        print(f"F1 Score for {model_name}: {f1:.4f}")
        print(f"{model_name} Accuracy: {accuracy:.4f}")
        print(f"{model_name} r2_score: {r2:.4f} Mae:{mae:.4f}")
        logging.info(f"{model_name} Accuracy: {accuracy:.4f}")
        logging.info(f"{model_name} r2_score: {r2:.4f} Mae:{mae:.4f}")
        logging.info(f"Confusion Matrix for {model_name}:\n{confusion_matri}\n")
        logging.info(f"Classification Report for {model_name}:\n{classification_report(y_test, y_pred)}\n")
        
        # Log params and metrics as before
        mlflow.log_params(model.get_params())
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("r2_score", r2)
        mlflow.log_metric("mean_squared_error", mae)
        mlflow.sklearn.log_model(model, "model")
        mlflow.sklearn.log_model(scaler, "scaler")
        mlflow.log_metric("precision",precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)
        logging.info(f"Model {model_name} logged to MLflow.")
        
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model = model
            best_model_run_id = mlflow.active_run().info.run_id # Get the run ID
            print(f"New best model found: {model_name} with accuracy {accuracy:.4f}")
            logging.info(f"New best model found: {model_name} with accuracy {accuracy:.4f}")



# After the loop
if best_model:
    # Register the best model in the MLflow Model Registry
    model_uri = f"runs:/{best_model_run_id}/model"
    mlflow.register_model(model_uri=model_uri, name="Iris-Classifier-Best")
    print("Best model has been registered in MLflow Model Registry.")



# --- NEW: Save the best model and scaler locally ---
output_dir = "saved_model"
if os.path.exists(output_dir):
    shutil.rmtree(output_dir) # Clean up old model files
os.makedirs(output_dir)

if best_model:
    joblib.dump(best_model, os.path.join(output_dir, "model.joblib"))
    joblib.dump(scaler, os.path.join(output_dir, "scaler.joblib"))
    print(f"\nBest model and scaler saved to '{output_dir}/' directory.")
    logging.info(f"Best model and scaler saved to '{output_dir}/' directory.")
else:
    print("No model was trained successfully.")
    logging.error("No model was trained successfully.")
logging.info("MLflow setup completed.")