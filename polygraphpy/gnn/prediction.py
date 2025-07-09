import pandas as pd
import numpy as np
import os
import torch
import logging
import torch
import math
import matplotlib.pyplot as plt

from tqdm import tqdm
from sklearn.metrics import mean_absolute_percentage_error, r2_score, mean_squared_error

from polygraphpy.gnn.models.gcn import GCN

logger = logging.getLogger("GNN")

class Prediction():
    def __init__(self, validation_data_path: str, gnn_output_path: str) -> None:
        self.validation_data_path = len(os.listdir(validation_data_path)) 
        self.gnn_output_path = gnn_output_path
        self.val_dataset = []
        self.model_hyperparameters = pd.read_csv(f'{gnn_output_path}/gcn_hyperparameters.csv')

        logger.info(f'Loading trained model.')
        self.model = GCN(self.model_hyperparameters['input_dim'].values[0], 
                            self.model_hyperparameters['conv_hidden_channels'].values[0], 
                            self.model_hyperparameters['mlp_hidden_channels'].values[0])

        logger.info(self.model)
        self.model = torch.load(f'{gnn_output_path}/model_gcn.pt')
        self.model.eval()

        logger.info(f'Reading validation data.')
        for i in range(self.validation_data_path):
            self.val_dataset.append(torch.load(f'{filedir}{i}.pt'))

    def make_prediction_with_val_data(self) -> None:
        pred = []
        y = []

        logger.info('Making prediction.')
        for graph in tqdm(self.val_dataset):
            y.append(graph.y.numpy()[0])
            out = self.model(graph.x, graph.edge_index, graph.edge_weight, torch.tensor(np.array([0])))
            pred.append(out.detach().numpy()[0][0])
        
        df_result = pd.DataFrame({'y': y, 'pred': pred})
        df_result = df_result.sort_values(by='y').reset_index(drop=True)

        self.make_plot(df_result)

        df_error = pd.DataFrame({'mape': round(mean_absolute_percentage_error(df_result.y.values,  df_result.pred.values)*100, 5),
                                 'r2': round(r2_score(df_result.y.values, df_result.pred.values), 5),
                                 'rmse': round(mean_squared_error(df_result.y.values, df_result.pred.values), 5)}, index=[0])
        
        logger.info(df_error)

        df_result.to_csv(f'{self.gnn_output_path}/df_results.csv')
        df_error.to_csv(f'{self.gnn_output_path}/df_error.csv')
        
        logger.info('Prediction done.')
    
    def make_plot(self, df_result: pd.DataFrame) -> None:
        p1 = [df_result.y.min(), df_result.y.max()]
        p2 = [df_result.y.min(), df_result.y.max()]

        coefficients = np.polyfit(p1, p2, 1)

        polynomial = np.poly1d(coefficients)
        x_axis = np.linspace(math.floor(df_result.y.min()), math.ceil(df_result.y.max()), 500)
        y_axis = polynomial(x_axis)

        fig = plt.figure()
        plt.scatter(df_result.y.values, df_result.pred.values)
        plt.plot(x_axis, y_axis, 'r')
        plt.xlim(df_result.y.min(), df_result.y.max())
        plt.ylim(df_result.y.min(), df_result.y.max())
        plt.grid(1)
        plt.xlabel("Ground Truths")
        plt.ylabel("Predictions")
        fig.savefig(f"{self.gnn_output_path}/pred.pdf", bbox_inches='tight')