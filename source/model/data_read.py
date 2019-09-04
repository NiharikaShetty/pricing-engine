import pandas as pd


class data_read:

    @staticmethod
    def read_csv_data(csv_path):
        return pd.read_csv(csv_path)