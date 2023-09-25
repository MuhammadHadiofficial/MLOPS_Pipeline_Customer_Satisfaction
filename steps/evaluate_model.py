import logging
import pandas as pd
from zenml import step
from sklearn.base import RegressorMixin
from src.evaluation import  MSE,RMSE,R2
from typing import Annotated,Tuple
from zenml.client import Client
import mlflow
experiment_tracker = Client().active_stack.experiment_tracker



@step(experiment_tracker=experiment_tracker.name)
def evaluate_model(model:RegressorMixin,X_test:pd.DataFrame,y_test:pd.Series)->Tuple[
    Annotated[float,"mse"],
     Annotated[float,"r2"],
      Annotated[float,"rmse"],
]:
    prediction=model.predict(X_test)
    mse=MSE()
    mse=mse.calculate_scores(prediction,y_test)
    mlflow.log_metric("mse", mse)
    r2=R2()
    
    r2=r2.calculate_scores(prediction,y_test)
    mlflow.log_metric("r2", r2)
    rmse=RMSE()
    rmse=rmse.calculate_scores(prediction,y_test)
    mlflow.log_metric("rmse", rmse)
    return mse,r2,rmse
