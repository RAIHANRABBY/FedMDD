import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import os

class DataLoader:
    def __init__(self, feature_path, label_path, num_clients, output_dir='data/clients'):
        self.feature_path = feature_path
        self.label_path = label_path
        self.num_clients = num_clients
        self.output_dir = output_dir

    def load_data(self):
        # Load features and labels
        feature_data = np.load(self.feature_path)
        self.data = np.moveaxis(feature_data['time_features'], 1, 2)  # Assuming 'time_features' is the key for data
        self.labels = np.load(self.label_path)['Y']

        # Normalize features
        scaler = MinMaxScaler()
        self.data = scaler.fit_transform(self.data.reshape(-1, self.data.shape[-1])).reshape(self.data.shape)

    def split_and_save_data(self):
        os.makedirs(self.output_dir, exist_ok=True)

        # Shuffle and split data for each client
        indices = np.arange(self.data.shape[0])
        np.random.shuffle(indices)
        data_shuffled = self.data[indices]
        labels_shuffled = self.labels[indices]

        data_splits = np.array_split(data_shuffled, self.num_clients)
        label_splits = np.array_split(labels_shuffled, self.num_clients)

        for client_id, (data, labels) in enumerate(zip(data_splits, label_splits)):
            x_train, x_test, y_train, y_test = train_test_split(
                data, labels, test_size=0.2, random_state=42
            )
            np.savez(
                os.path.join(self.output_dir, f"client_{client_id}.npz"),
                x_train=x_train,
                y_train=y_train,
                x_test=x_test,
                y_test=y_test
            )
        print(f"Data for {self.num_clients} clients saved in '{self.output_dir}'.")



if __name__ =="__main__":
        # Create an instance of DataLoader
    data_loader = DataLoader(
        feature_path='E:/MDD/data/eeg_features.npz',
        label_path='E:/MDD/data/Combined_labels.npz',
        num_clients=10
    )
    
    # Load the data before splitting
    data_loader.load_data()
    
    # Split and save the data for each client
    data_loader.split_and_save_data()