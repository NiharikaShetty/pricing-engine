import lightgbm as lgb
import matplotlib.pyplot as plt
import  pandas as pd
from math import sqrt
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble.bagging import BaggingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, recall_score, mean_squared_error
from xgboost import XGBRegressor


class Regression:
    def __init__(self,x_train,y_train):
        self.x_train = x_train
        self.y_train = y_train

    def precision(self, y_true, y_predict):
        precision_result = []
        y_true = list(y_true)
        y_predict = list(y_predict)
        if len(y_true) == len(y_predict):
            for i in range(len(y_predict)):
                result = 0
                if y_predict[i] >= y_true[i]:
                    result = y_true[i] / y_predict[i]
                else:
                    result = y_predict[i] / y_true[i]
                precision_result.append(result)
        # print(precision_result)
        return sum(precision_result) / len(precision_result)

    def score_r2_precision(self, model, x_test, y_test):
        score = '{0:.2f}'.format(model.score(x_test, y_test))
        prediction = model.predict(x_test).astype(int)
        y_true, y_pred = list(y_test.values.astype(int)), list(prediction)
        r2score = '{0:.2f}'.format(r2_score(y_pred,y_true))
        recol = '{0:.2f}'.format(recall_score(y_true, y_pred, average='weighted'))
        pre = '{0:.2f}'.format(self.precision(y_true, y_pred))
        rms = '{0:.2f}'.format(sqrt(mean_squared_error(y_true, y_pred)))
        # print('Intercept: \n', model.intercept_)
        return {'Score':score,'r2_score':r2score,'precision':pre,'prediction':prediction, 'recol':recol, 'rms':rms}

    def fiting_model(self, model):
        return model.fit(self.x_train, self.y_train)

    def xboost_regression(self):
        model = XGBRegressor()
        return self.fiting_model(model)

    def light_gbm_regression(self):
        model = lgb.LGBMRegressor()
        return self.fiting_model(model)

    def linear_regression(self):
        model = LinearRegression(normalize=True)
        return self.fiting_model(model)

    def random_forest_regression(self):
        model = RandomForestRegressor()
        return self.fiting_model(model)

    def gredient_boosting(self):
        model =  GradientBoostingRegressor()
        return self.fiting_model(model)

    def plot_graph(self, x, y, x_label, y_label):
        plt.title(x_label + ' vs ' + y_label)
        plt.scatter(x, y)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.show()


