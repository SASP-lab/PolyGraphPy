import pandas as pd
import torch

from random import shuffle
from tqdm import tqdm
from torch_geometric.loader import DataLoader

from polygraphpy.gnn.models.gcn import GCN

class Train():
    def __init__(self, conv_hidden_channels:int, mlp_hidden_channels:int, data: pd.DataFrame, learning_rate: float, batch_size: int = 8, epochs: int = 100,
                 train_input_data_path: str = None, gnn_output_path: str = None, validation_data_path: str = None) -> None:
        self.training_dataset = []
        self.input_dim = 0
        self.min_val_error = 10e6
        self.train_input_data_path = train_input_data_path
        self.gnn_output_path = gnn_output_path
        self.validation_data_path = validation_data_path
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.epochs = epochs

        self.read_train_data(data)

        self.training_model = GCN(self.input_dim, conv_hidden_channels, mlp_hidden_channels)

        print('Model architecture:')
        print(self.training_model)

        self.model_hyperparameters = pd.DataFrame({'input_dim': self.input_dim,
                                                    'conv_hidden_channels': conv_hidden_channels,
                                                    'mlp_hidden_channels': mlp_hidden_channels}, index=[0])
        
        self.model_hyperparameters.to_csv(f'{gnn_output_path}/gcn_hyperparameters.csv', index=False)

        self.optimizer = torch.optim.Adam(self.training_model.parameters(), lr=learning_rate)
        self.criterion = torch.nn.MSELoss()
    
    def read_train_data(self, data: pd.DataFrame) -> None:
        print(f'Reading training data.')
        for row in tqdm(data.itertuples()):
            self.training_dataset.append(torch.load(f'{self.train_input_data_path}/{row.id}.pt', weights_only=False))
        
        self.input_dim = self.training_dataset[0].x.shape[1]

    def create_train_and_validation_dataset(self) -> tuple[list, list]:
        print(f'Spliting data into training dataset and validation dataset.')
        shuffle(self.training_dataset)

        dist = 0.90

        dataset_len = len(self.training_dataset)

        train_dataset = self.training_dataset[:int(dataset_len*dist)]
        val_dataset = self.training_dataset[int(dataset_len*dist):]

        print(f'Number of training graphs: {len(train_dataset)}')
        print(f'Number of validation graphs: {len(val_dataset)}')

        return train_dataset, val_dataset

    def create_batches(self, train_dataset: list, val_dataset: list, batch_size: int) -> tuple[list, list]:
        print(f'Creating batches.')
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

        return train_loader, val_loader

    def train_model(self, train_loader: DataLoader):
        self.training_model.train()
        k = 0
        error = 0

        for data in train_loader:
            self.optimizer.zero_grad()
            out = self.training_model(data.x, data.edge_index, data.edge_weight, data.batch)
            loss = self.criterion(out.reshape(len(out)), data.y)
            loss.backward()
            self.optimizer.step()
            error = error + loss.item()
            k+=1
        error = error/k

        return error

    def model_validation(self, val_loader: DataLoader, epoch: int):
        self.training_model.eval()
        k = 0
        error = 0

        for data in val_loader:
            out = self.training_model(data.x, data.edge_index, data.edge_weight, data.batch)
            loss = self.criterion(out.reshape(len(out)), data.y)
            error = error + loss.item()
            k+=1
        error = error/k

        if (error < self.min_val_error):
            print(f'Model updated with best result: Val loss = {error:.5f} at epoch = {epoch}')
            torch.save(self.training_model, f'{self.gnn_output_path}/model_gcn.pt')
            self.min_val_error = error
        
        return error
    
    def save_validation_data(self, val_dataset: list):
        print(f'Saving validation data.')

        i = 0
        for graph in tqdm(val_dataset):
            torch.save(graph, f'{self.validation_data_path}/{graph.mol_id}.pt')
            i+=1
    
    def run(self):
        train_dataset, val_dataset = self.create_train_and_validation_dataset()

        train_loader, val_loader = self.create_batches(train_dataset, val_dataset, self.batch_size)

        print(f'Starting training and validation with epochs = {self.epochs} and learning rate = {self.learning_rate}.')
        for epoch in range(self.epochs):
            loss = self.train_model(train_loader)
            val_loss = self.model_validation(val_loader, epoch)

            print(f'Epoch: {epoch}, Train Loss: {loss:.5f}, Val Error: {val_loss:.5f}')

        self.save_validation_data(val_dataset)
        
        print(f'Train finished.')