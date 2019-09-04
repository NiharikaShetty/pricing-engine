
class Constants:
    TRAIN_TEST_SPLIT = 0.2
    CSV_PATH = 'source/dataSource/ivv.csv'
    PREDICTION_TARGET = 'SalesPrice'
    PREDICTION_COLUMNS = ['Mileage', 'Color', 'Year', 'MMRAvgValue', 'ModelName', 'MakeName', 'StyleName', 'AddrZip','SalesPrice']
    ENCODE_COLUMNS = ['MakeName', 'ModelName', 'StyleName', 'AddrZip', 'Color']