from zenml import pipeline,step
from zenml.config import DockerSettings
from zenml.constants import  DEFAULT_SERVICE_START_STOP_TIMEOUT
from zenml.integrations.constants import MLFLOW
from zenml.integrations.mlflow.model_deployers.mlflow_model_deployer import (MLFlowModelDeployer)
from zenml.integrations.mlflow.services.mlflow_deployment import MLFlowDeploymentService
from zenml.steps import BaseParameters, Output
from steps.clean_data import clean_data
from steps.evaluate_model import  evaluate_model
from steps.ingest_data import  ingest_data
from steps.model_train import train_model
docker_settings=DockerSettings(required_integrations=[MLFLOW])
@pipeline(enable_cache=True,settings={
    "docker_settings":docker_settings
})
# def continous_deployment_pipeline():