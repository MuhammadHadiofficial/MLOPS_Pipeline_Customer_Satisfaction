import logging 
from zenml import step
import pandas as pd

class IngestData:
    """
    Ingesting data from data_path
    """
    def __init__(self,data_path:str) -> None:
        """
        Args:
        """
        self.data_path=data_path
    def get_data(self):
        """Ingest data from data path """
        logging.info("Ingetsting data from {self.data_path}")
        return pd.read_csv(self.data_path)
    
@step
def ingest_data(data_path:str)->pd.DataFrame:
    """
    Ingesting the data from the data path.

    Args:
    data_path: path to the data
    Returns:
        pd.DataFrame: the ingested data
    """
    try:
        ingest_data=IngestData(data_path)
        df=ingest_data.get_data()
        return df
    except Exception as e:
        logging.error("Error while ingesting data: {e}")
