import logging
from pipelines.training_pipeline import  training_pipeline
import os
from zenml.integrations.mlflow.mlflow_utils import get_tracking_uri
if __name__=="__main__":
    print(get_tracking_uri())
    logging.info(os.path.join( os.path.realpath(__file__),'../data/olist_customers_dataset.csv'))
    training_pipeline('D:\learning_2022\mlops_customer_satisfaction\data\olist_customers_dataset.csv')
    print(
        "Now run \n "
        f"    mlflow ui --backend-store-uri '{get_tracking_uri()}'\n"
        "To inspect your experiment runs within the mlflow UI.\n"
        "You can find your runs tracked within the `mlflow_example_pipeline`"
        "experiment. Here you'll also be able to compare the two runs.)"
    )