"""
polygraphpy.gnn.prediction
==========================

This module provides the `Prediction` class for evaluating a trained Graph
Neural Network (GNN) model. It handles loading a saved model and a validation
dataset, performing predictions on the validation set, and calculating
key performance metrics such as R-squared ($R^2$), Mean Squared Error (MSE),
and Mean Absolute Percentage Error (MAPE). The results are saved to CSV
files, and a parity plot is generated for visual analysis.
"""

import pandas as pd
import numpy as np
import os
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import mean_absolute_percentage_error, r2_score, mean_squared_error

class Prediction():
    """Evaluates a trained GNN model and reports its performance.

    This class loads a pre-trained GNN model and a validation dataset. It then
    uses the model to make predictions on the validation data, calculates
    common regression metrics, and generates a parity plot to compare
    predictions against ground truth values.

    :param validation_data_path: Directory where the validation dataset (`.pt` files) is stored.
    :type validation_data_path: str
    :param gnn_output_path: Directory where the trained model and output files (metrics, plots)
                            are to be saved.
    :type gnn_output_path: str
    :param polymer_type: The type of polymer the model was trained on ('monomer' or 'copolymer').
    :type polymer_type: str
    """
    def __init__(self, validation_data_path: str, gnn_output_path: str, polymer_type: str) -> None:
        """Initializes the Prediction class by loading the model and data.
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f'Using device: {self.device}')

        self.validation_data = os.listdir(validation_data_path)
        self.gnn_output_path = gnn_output_path
        self.val_dataset = []
        self.polymer_type = polymer_type

        print(f'Loading trained model.')
        if self.polymer_type == 'copolymer':
            self.model = torch.load(f'{gnn_output_path}model_gnn_copoly.pt', weights_only=False)
        else:
            self.model = torch.load(f'{gnn_output_path}model_gcn.pt', weights_only=False)
        print(self.model)
        self.model.to(self.device)
        self.model.eval()

        print(f'Reading validation data.')
        for i in self.validation_data:
            self.val_dataset.append(torch.load(f'{validation_data_path}/{i}', weights_only=False))

    def make_plot(self, df_result: pd.DataFrame) -> None:
        """Generates and saves a parity plot of ground truths vs. predictions.

        The plot includes a red line representing the ideal 1:1 correlation
        between ground truth and predictions.

        :param df_result: A DataFrame with columns 'y' (ground truth) and 'pred' (prediction).
        :type df_result: pd.DataFrame
        :return: None
        :rtype: None
        """
        p1 = [0, 1]
        p2 = [0, 1]

        coefficients = np.polyfit(p1, p2, 1)

        polynomial = np.poly1d(coefficients)
        x_axis = np.linspace(0, 1, 500)
        y_axis = polynomial(x_axis)

        fig = plt.figure()
        plt.scatter(df_result.y.values, df_result.pred.values)
        plt.plot(x_axis, y_axis, 'r')
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.grid(1)
        plt.xlabel("Ground Truths")
        plt.ylabel("Predictions")
        fig.savefig(f"{self.gnn_output_path}/pred.pdf", bbox_inches='tight')

    def run(self) -> None:
        """Executes the prediction and evaluation pipeline.

        This is the main method that performs the following steps:
        1. Iterates through the validation dataset to get predictions.
        2. Compiles ground truth and prediction values into a DataFrame.
        3. Calculates performance metrics (MAPE, R2, MSE).
        4. Saves the results and metrics to CSV files.
        5. Generates and saves a parity plot.

        :return: None
        :rtype: None
        """
        pred = []
        y = []
        print('Making prediction.')
        for graph in tqdm(self.val_dataset):
            y.append(graph.y.numpy()[0])
            graph = graph.to(self.device)
            # Ensure batch tensor matches the number of nodes in graph.x
            batch = torch.zeros(graph.x.size(0), dtype=torch.long).to(self.device)  # Single graph, all nodes assigned to index 0
            out = self.model(graph.x, graph.edge_index, graph.edge_weight, batch)
            pred.append(out.detach().cpu().numpy()[0][0])
        
        df_result = pd.DataFrame({'y': y, 'pred': pred})
        df_result = df_result.sort_values(by='y').reset_index(drop=True)
        self.make_plot(df_result)
        
        df_error = pd.DataFrame({
            'mape': round(mean_absolute_percentage_error(df_result.y.values, df_result.pred.values) * 100, 5),
            'r2': round(r2_score(df_result.y.values, df_result.pred.values), 5),
            'mse': round(mean_squared_error(df_result.y.values, df_result.pred.values), 5)
        }, index=[0])
        print(df_error)
        
        df_result.to_csv(f'{self.gnn_output_path}/df_results.csv')
        df_error.to_csv(f'{self.gnn_output_path}/df_error.csv')

        print('Prediction done.')