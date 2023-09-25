from zenml import pipeline
from steps.ingest_data import ingest_data
from steps.clean_data import clean_data
from steps.model_train import train_model
from steps.evaluate_model import evaluate_model

@pipeline()
def training_pipeline(data_path:str):
    df=ingest_data(data_path)
    X_train,X_test,y_train,y_test=clean_data(df)

    model=train_model(X_train,X_test,y_train,y_test)
    evaluate_model(model,X_test,y_test)