from sklearn.model_selection import train_test_split

from source.interface.Prediction import Prediction
from source.model.data_read import data_read
from source.model.regression import Regression
from source.model.encodeData import encodeData, Constants


class prediction_manager(Prediction):
    def __init__(self):
        pass

    def get_predicted_price(self):
        data = self.read_data()
        data = data[Constants.PREDICTION_COLUMNS].copy()
        data = self.encode_data(data)
        self.x_train, self.x_test, self.y_train, self.y_test = self.split_data(data)
        regression = Regression(self.x_train,self.y_test)
        model = regression.random_forest_regression()
        return regression.score_r2_precision(model,self.x_test,self.y_test)


    def read_data(self):
        return data_read.read_csv_data(Constants.CSV_PATH)

    def encode_data(self, data):
        encode_data = encodeData(data)
        return encode_data.backward_difference(Constants.ENCODE_COLUMNS)

    # Method to split data.
    def split_data(self, data):

        return train_test_split(data.drop([Constants.PREDICTION_TARGET], axis=1),
                                                            data.loc[:, Constants.PREDICTION_TARGET],
                                                            test_size=Constants.TRAIN_TEST_SPLIT,
                                                            random_state=42)
