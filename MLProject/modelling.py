import argparse
import os
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.svm import SVC
import numpy as np

def main(data_dir):
    mlflow_tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    mlflow_username     = os.getenv("MLFLOW_TRACKING_USERNAME")
    mlflow_password     = os.getenv("MLFLOW_TRACKING_PASSWORD")

    if not mlflow_tracking_uri or not mlflow_username or not mlflow_password:
        raise EnvironmentError("MLFLOW_TRACKING_URI, MLFLOW_TRACKING_USERNAME, dan MLFLOW_TRACKING_PASSWORD harus di-set sebagai environment variable")

    # Set environment variables untuk autentikasi MLflow
    os.environ["MLFLOW_TRACKING_USERNAME"] = mlflow_username
    os.environ["MLFLOW_TRACKING_PASSWORD"] = mlflow_password

    # Set tracking URI dan experiment
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    mlflow.set_experiment("Diabetes Modeling - Hyperparameter Tuning")

    # Aktifkan autolog MLflow untuk sklearn
    mlflow.sklearn.autolog()

    # Load data
    X_train = pd.read_csv(os.path.join(data_dir, "x_train.csv"))
    X_test = pd.read_csv(os.path.join(data_dir, "x_test.csv"))
    y_train = pd.read_csv(os.path.join(data_dir, "y_train.csv")).squeeze()
    y_test = pd.read_csv(os.path.join(data_dir, "y_test.csv")).squeeze()

    input_example = X_train.head(5)

    C_range = np.logspace(-2, 2, 5)
    kernel_options = ['linear', 'rbf', 'poly']
    gamma_range = ['scale', 'auto']

    best_accuracy = 0
    best_params = {}
    best_model = None 

    with mlflow.start_run(): 
        for C in C_range:
            for kernel in kernel_options:
                for gamma in gamma_range:
                    with mlflow.start_run(run_name=f"SVC_C{C}_kernel{kernel}_gamma{gamma}", nested=True):
                        model = SVC(C=C, kernel=kernel, gamma=gamma)
                        model.fit(X_train, y_train)
                        accuracy = model.score(X_test, y_test)
                        mlflow.log_metric("accuracy", accuracy)

                        if accuracy > best_accuracy:
                            best_accuracy = accuracy
                            best_params = {"C": C, "kernel": kernel, "gamma": gamma}
                            best_model = model 
        print("Best Accuracy:", best_accuracy)
        print("Best Params:", best_params)
        mlflow.log_metric("best_accuracy", best_accuracy)
        mlflow.log_params(best_params)

        if best_model:
            mlflow.sklearn.log_model(
                sk_model=best_model,
                artifact_path="best_model",
                input_example=input_example
            )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="dataset_preprocessing", help="Path ke folder data")
    args = parser.parse_args()

    main(args.data_dir)
