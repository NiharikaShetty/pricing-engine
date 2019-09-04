import abc


# This is the pricing engine interface
# It defines all the compulsory methods.
class Prediction(abc.ABC):
    @abc.abstractmethod
    def get_predicted_price(self):
        pass

    @abc.abstractmethod
    def read_data(self):
        pass

    @abc.abstractmethod
    def encode_data(self,data):
        pass

