from category_encoders import *
from sklearn import preprocessing
from source.utility.Constants import Constants


class encodeData:
    def __init__(self, data):
        self.data = data
        self.y = data[Constants.PREDICTION_TARGET]
        self.x = data.drop([Constants.PREDICTION_TARGET], axis=1)

    def marge_data(self,data):
        data[Constants.PREDICTION_TARGET] = self.y
        return data

    def backward_difference(self,column):
        encode = BackwardDifferenceEncoder(cols=column, drop_invariant=True)
        encode_data = encode.fit_transform(self.x, self.y)
        return self.marge_data(encode_data)

    def one_hot_encode(self,column,):
        encode = OneHotEncoder(cols=column, drop_invariant=True).fit_transform(self.x, self.y)
        return self.marge_data(encode)

    def binary_encode(self, column):
        encode = BinaryEncoder(cols=column, drop_invariant=True).fit_transform(self.x, self.y)
        return self.marge_data(encode)

    def sum_encode(self,column):
        encode = SumEncoder(cols=column, drop_invariant=True).fit_transform(self.x, self.y)
        return self.marge_data(encode)

    def polinomial_encode(self,column):
        encode = PolynomialEncoder(cols=column, drop_invariant=True).fit_transform(self.x, self.y)
        return self.marge_data(encode)

    def helmert_encode(self,column):
        encode = HelmertEncoder(cols=column, drop_invariant=True).fit_transform(self.x, self.y)
        return self.marge_data(encode)

    def base_n_encode(self,column):
        encode = BaseNEncoder(cols=column, drop_invariant=True).fit_transform(self.x, self.y)
        return self.marge_data(encode)

    def normal_encode(self, column):
        df = self.data
        for c in column:
            df[c] = preprocessing.LabelEncoder().fit_transform(df[c].astype(str))
        return self.marge_data(df)
