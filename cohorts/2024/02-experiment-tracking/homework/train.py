import os
import pickle
import click

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

import mlflow
from mlflow import MlflowClient

def load_pickle(filename: str):
    with open(filename, "rb") as f_in:
        return pickle.load(f_in)


# @click.command()
# @click.option(
#     "--data_path",
#     default="./output",
#     help="Location where the processed NYC taxi trip data was saved"
# )
def run_train(data_path: str):
    # enable autologging
    mlflow.sklearn.autolog()
    
    X_train, y_train = load_pickle(os.path.join(data_path, "train.pkl"))
    X_val, y_val = load_pickle(os.path.join(data_path, "val.pkl"))

    with mlflow.start_run() as run:
        max_depth = 10
        
        rf = RandomForestRegressor(max_depth=max_depth, random_state=0)
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_val)

        rmse = mean_squared_error(y_val, y_pred, squared=False)
    
    return run


if __name__ == '__main__':
    MLFLOW_TRACKING_URI = "sqlite:///mlflow.db"

    client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)
    # client.search_experiments()
    run_train()
