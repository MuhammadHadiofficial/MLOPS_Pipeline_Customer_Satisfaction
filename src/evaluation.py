import logging
import numpy as np
from sklearn.metrics import mean_squared_error,r2_score
from abc import ABC,abstractmethod
class Evaluation(ABC):
    @abstractmethod
    def calculate_scores(self,y_true:np.ndarray,y_pred:np.ndarray):
        pass

class MSE(Evaluation):
    def calculate_scores(self, y_true: np.ndarray, y_pred: np.ndarray):
        try:
            logging.info("Calculating MSE")
            score=mean_squared_error(y_true,y_pred)
            logging.info("MSE:{}".format(score))
            return score
        except Exception as e:
            logging.error(" Error in calculating MSE: {}".format(e))
            raise e
class R2(Evaluation):
    
    def calculate_scores(self, y_true: np.ndarray, y_pred: np.ndarray):
        try:
            logging.info("Calculating MSE")
            score=r2_score(y_true,y_pred)
            logging.info("r2_score:{}".format(score))
            return score
        except Exception as e:
            logging.error(" Error in calculating r2_score: {}".format(e))
            raise e
class RMSE(Evaluation):
    def calculate_scores(self, y_true: np.ndarray, y_pred: np.ndarray):
        try:
            logging.info("Calculating MSE")
            score=mean_squared_error(y_true,y_pred,squared=False)
            logging.info("RMSE:{}".format(score))
            return score
        except Exception as e:
            logging.error(" Error in calculating MSE: {}".format(e))
            raise e       