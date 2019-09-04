from source.controller.prediction_manager import prediction_manager
prediction = prediction_manager()
result = prediction.get_predicted_price()
print(result)