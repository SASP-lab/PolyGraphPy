import pandas as pd

class PreProcess():
    def __init__(self, input_csv: str = None, train_input_data_path: str = None):
        self.input_csv = input_csv
        self.train_input_data_path = train_input_data_path

        self.df = pd.read_csv(self.input_csv)

    def run(self):
        pass