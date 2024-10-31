import flwr as fl
from task import load_model
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    cohen_kappa_score,
    roc_auc_score,
    confusion_matrix
)

class FederatedClient(fl.client.NumPyClient):
    def __init__(self, model, client_data, client_id):
        self.model = model
        self.client_data = client_data
        self.client_id = client_id

    def get_parameters(self):
        return self.model.get_weights()

    def fit(self, parameters, config):
        self.model.set_weights(parameters)
        x_train, y_train = self.client_data['x_train'], self.client_data['y_train']
        self.model.fit(x_train, y_train, epochs=20, batch_size=32, verbose=1)
        return self.model.get_weights(), len(x_train), {}

    def evaluate(self, parameters, config):
        self.model.set_weights(parameters)
        x_test, y_test = self.client_data['x_test'], self.client_data['y_test']
        
        # Get model predictions
        y_pred = self.model.predict(x_test).round()  # Assuming binary classification
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)  # Sensitivity
        f1 = f1_score(y_test, y_pred)
        kappa = cohen_kappa_score(y_test, y_pred)
        
        # Calculate specificity
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        specificity = tn / (tn + fp)
        
        # Calculate AUC-ROC
        auc_roc = roc_auc_score(y_test, y_pred)
        
        # Calculate loss (for demonstration, replace with actual loss calculation if needed)
        loss, accuracy = self.model.evaluate(x_test, y_test, verbose=0)

        # Create a DataFrame to store evaluation results
        evaluation_data = {
            "client_id": [self.client_id],
            "loss": [loss],
            "accuracy": [accuracy],
            "precision": [precision],
            "recall": [recall],
            "f1_score": [f1],
            "specificity": [specificity],
            "kappa": [kappa],
            "auc_roc": [auc_roc]
        }
        
        df = pd.DataFrame(evaluation_data)
        
        # Save the DataFrame to CSV, appending without headers after the first write
        csv_file = "evaluation_results.csv"
        df.to_csv(csv_file, mode="a", index=False, header=not pd.io.common.file_exists(csv_file))

        # Return metrics for federated learning evaluation
        return loss, len(x_test), {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "specificity": specificity,
            "kappa": kappa,
            "auc_roc": auc_roc
        }

def load_client_data(client_id):
    data_path = f"data/clients/client_{client_id}.npz"
    with np.load(data_path) as data:
        return {
            'x_train': data['x_train'],
            'y_train': data['y_train'],
            'x_test': data['x_test'],
            'y_test': data['y_test']
        }

def client_fn(client_id: int) -> fl.client.Client:
    model = load_model()
    client_data = load_client_data(client_id)
    client = FederatedClient(model, client_data, client_id).to_client()
    return client

def start_client(client_id):
    fl.client.start_client(
        server_address="127.0.0.1:8080",
        client=client_fn(client_id)
    )

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        client_id = int(sys.argv[1])
        start_client(client_id)
    else:
        print("Please provide a client_id as an argument.")
