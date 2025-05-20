# Structure

### preprocess_data.py

Takes training, validation and test sets from data folder and preprocesses them for training

### train.py

Fitting a randomforest regressor on the data and using autologging

### hpo.py

Doing hyper-parameter optimization using the validation set and tracking rmse using mlflow

### register_model.py

Compares the top 5 performing models on the test set then registers the best one using mlflow
