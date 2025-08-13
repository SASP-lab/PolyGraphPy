"""
polygraphpy.gnn.train
=====================

This module provides the `Train` class for training a Graph Neural Network
(GNN) model. It handles the complete training and validation pipeline,
including reading prepared graph data, splitting the dataset, creating
data loaders, and running the training and evaluation loops over a specified
number of epochs. It also saves the best-performing model and training
statistics.
"""

import pandas as pd
import torch

from random import shuffle
from tqdm import tqdm
from torch_geometric.loader import DataLoader
from polygraphpy.gnn.models.gcn import GCN
from polygraphpy.gnn.models.graphunet import GraphUNetModel

class Train():
    """Manages the training and validation of a GNN model.

    This class orchestrates the entire training process. It loads pre-processed
    graph data, splits it into training and validation sets, initializes a
    GNN model (GCN or GraphUNet), defines the optimizer and loss function,
    and then runs the training loop to minimize the model's error. The best
    model based on validation loss is saved.

    :param conv_hidden_channels: Number of hidden channels in the convolutional layers.
    :type conv_hidden_channels: int
    :param mlp_hidden_channels: Number of hidden channels in the MLP layers.
    :type mlp_hidden_channels: int
    :param data: The pandas DataFrame containing the pre-processed molecular data.
    :type data: pd.DataFrame
    :param learning_rate: The learning rate for the Adam optimizer.
    :type learning_rate: float
    :param batch_size: The number of graphs per batch. Defaults to 8.
    :type batch_size: int, optional
    :param epochs: The number of training epochs. Defaults to 100.
    :type epochs: int, optional
    :param train_input_data_path: Directory to load the PyTorch Geometric graph data files from.
                                  Defaults to None.
    :type train_input_data_path: str, optional
    :param gnn_output_path: Directory to save the trained models and training statistics.
                            Defaults to None.
    :type gnn_output_path: str, optional
    :param validation_data_path: Directory to save the validation dataset files.
                                 Defaults to None.
    :type validation_data_path: str, optional
    :param polymer_type: The type of polymer being trained ('monomer' or 'copolymer').
                         Defaults to 'monomer'.
    :type polymer_type: str, optional
    :param model: The name of the GNN model to use ('gcn' or 'graphunet').
                  Defaults to 'gcn'.
    :type model: str, optional
    """
    def __init__(self, conv_hidden_channels:int, mlp_hidden_channels:int, data: pd.DataFrame, learning_rate: float, batch_size: int = 8, epochs: int = 100,
                 train_input_data_path: str = None, gnn_output_path: str = None, validation_data_path: str = None, polymer_type: str = 'monomer',
                 model: str = 'gcn') -> None:
        """Initializes the training class with model and data parameters.
        """
        self.training_dataset = []
        self.input_dim = 0
        self.min_val_error = 10e9
        self.train_input_data_path = train_input_data_path
        self.gnn_output_path = gnn_output_path
        self.validation_data_path = validation_data_path
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.polymer_type = polymer_type

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f'Device: {self.device}')

        self.read_train_data(data)

        if (model == 'gcn'):
            self.training_model = GCN(self.input_dim, conv_hidden_channels, mlp_hidden_channels)
        else:
            self.training_model = GraphUNetModel(self.input_dim, conv_hidden_channels, mlp_hidden_channels)
        self.training_model = self.training_model.to(self.device)

        print('Model architecture:')
        print(self.training_model)

        self.optimizer = torch.optim.Adam(self.training_model.parameters(), lr=learning_rate)
        self.criterion = torch.nn.MSELoss()
    
    def read_train_data(self, data: pd.DataFrame) -> None:
        """Reads the PyTorch Geometric graph data from files into a list.

        :param data: The pandas DataFrame containing metadata about the graphs to be loaded.
        :type data: pd.DataFrame
        :return: None
        :rtype: None
        """
        print(f'Reading training data. Size: {len(data)}')
        for row in tqdm(data.itertuples()):
            try:
                if self.polymer_type != 'copolymer':
                    self.training_dataset.append(torch.load(f'{self.train_input_data_path}{row.id_A}_{row.chain_size}.pt', weights_only=False))
                else:
                    self.training_dataset.append(torch.load(f'{self.train_input_data_path}{row.id_A}_{row.id_B}_{row.chain_size}.pt', weights_only=False))
            except:
                continue
        
        self.input_dim = self.training_dataset[0].x.shape[1]

    def create_train_and_validation_dataset(self) -> tuple[list, list]:
        """Splits the full dataset into training and validation sets.

        The split is a fixed 90/10 ratio, and the data is shuffled beforehand.

        :return: A tuple containing the training dataset and the validation dataset.
        :rtype: tuple[list, list]
        """
        print(f'Spliting data into training dataset and validation dataset.')
        shuffle(self.training_dataset)

        dist = 0.90

        dataset_len = len(self.training_dataset)

        train_dataset = self.training_dataset[:int(dataset_len*dist)]
        val_dataset = self.training_dataset[int(dataset_len*dist):]

        print(f'Number of training graphs: {len(train_dataset)}')
        print(f'Number of validation graphs: {len(val_dataset)}')

        return train_dataset, val_dataset

    def create_batches(self, train_dataset: list, val_dataset: list, batch_size: int) -> tuple[DataLoader, DataLoader]:
        """Creates PyTorch Geometric DataLoaders for training and validation.

        :param train_dataset: The list of graphs for training.
        :type train_dataset: list
        :param val_dataset: The list of graphs for validation.
        :type val_dataset: list
        :param batch_size: The number of graphs per batch.
        :type batch_size: int
        :return: A tuple containing the training DataLoader and the validation DataLoader.
        :rtype: tuple[DataLoader, DataLoader]
        """
        print(f'Creating batches.')
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

        return train_loader, val_loader

    def train_model(self, train_loader: DataLoader) -> float:
        """Runs a single epoch of the training loop.

        :param train_loader: The DataLoader for the training set.
        :type train_loader: DataLoader
        :return: The average training loss for the epoch.
        :rtype: float
        """
        self.training_model.train()
        total_loss = 0
        total_samples = 0

        for data in train_loader:
            data = data.to(self.device)
            self.optimizer.zero_grad()
            out = self.training_model(data.x, data.edge_index, data.edge_weight, data.batch)
            loss = self.criterion(out.reshape(len(out)), data.y)
            loss.backward()
            self.optimizer.step()
            batch_size = data.y.size(0)  # Number of samples in the batch
            total_loss += loss.item() * batch_size  # Weight loss by batch size
            total_samples += batch_size

        return total_loss / total_samples if total_samples > 0 else 0

    def model_validation(self, val_loader: DataLoader, epoch: int) -> float:
        """Runs a single validation loop and saves the best model.

        :param val_loader: The DataLoader for the validation set.
        :type val_loader: DataLoader
        :param epoch: The current epoch number.
        :type epoch: int
        :return: The average validation loss for the epoch.
        :rtype: float
        """
        self.training_model.eval()
        total_loss = 0
        total_samples = 0

        for data in val_loader:
            data = data.to(self.device)
            out = self.training_model(data.x, data.edge_index, data.edge_weight, data.batch)
            loss = self.criterion(out.reshape(len(out)), data.y)
            batch_size = data.y.size(0)  # Number of samples in the batch
            total_loss += loss.item() * batch_size  # Weight loss by batch size
            total_samples += batch_size

        avg_loss = total_loss / total_samples if total_samples > 0 else 0

        if avg_loss < self.min_val_error:
            print(f'Model updated with best result: Val loss = {avg_loss:.5f} at epoch = {epoch}')
            if self.polymer_type == 'copolymer':
                torch.save(self.training_model, f'{self.gnn_output_path}model_gnn_copoly.pt')
            else:
                torch.save(self.training_model, f'{self.gnn_output_path}model_gcn.pt')
            self.min_val_error = avg_loss
        
        return avg_loss
    
    def save_validation_data(self, val_dataset: list):
        """Saves the validation dataset to a specified folder.

        :param val_dataset: The list of graphs in the validation set.
        :type val_dataset: list
        :return: None
        :rtype: None
        """
        print(f'Saving validation data.')

        for graph in tqdm(val_dataset):
            if self.polymer_type != 'copolymer':
                torch.save(graph, f'{self.validation_data_path}{int(graph.mol_id.detach().numpy()[0])}_{int(graph.chain_size.detach().numpy()[0])}.pt')
            else:
                torch.save(graph, f'{self.validation_data_path}{int(graph.id_A.detach().numpy()[0])}_{int(graph.id_B.detach().numpy()[0])}_{int(graph.chain_size.detach().numpy()[0])}.pt')

    def save_training_statistics(self, df: pd.DataFrame):
        """Saves the training and validation loss per epoch to a CSV file.

        :param df: The DataFrame containing the training statistics.
        :type df: pd.DataFrame
        :return: None
        :rtype: None
        """
        aux = ''

        if self.polymer_type == 'copolymer':
            aux = '_copoly'

        df.to_csv(f'{self.gnn_output_path}training_statistics{aux}.csv', index=False)
    
    def run(self):
        """Executes the full training and validation pipeline.

        This is the main public method that orchestrates the entire process,
        from splitting the data to running the epoch loop and saving the
        results.

        :return: None
        :rtype: None
        """
        df_train_statistics = pd.DataFrame()

        train_dataset, val_dataset = self.create_train_and_validation_dataset()
        self.save_validation_data(val_dataset)
        
        train_loader, val_loader = self.create_batches(train_dataset, val_dataset, self.batch_size)

        print(f'Starting training and validation with epochs = {self.epochs} and learning rate = {self.learning_rate}.')
        for epoch in range(self.epochs):
            loss = self.train_model(train_loader)
            val_loss = self.model_validation(val_loader, epoch)

            print(f'Epoch: {epoch}, Train Loss: {loss:.5f}, Val Error: {val_loss:.5f}')

            df_train_statistics = pd.concat([df_train_statistics, pd.DataFrame({'epoch': epoch, 'train_loss': loss, 'val_loss': val_loss}, index=[0])])

        self.save_training_statistics(df_train_statistics)
        
        print(f'Train finished.')