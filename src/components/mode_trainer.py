import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression,Ridge,Lasso,ElasticNet
from src.exception import CustomException
from src.logger import logging

from src.utils import save_object
from src.utils import evaluate_model 

from dataclasses import dataclass
import sys
import os

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts','model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def inititate_model_training(self,train_array,test_array):
        try:
            logging.info("Splitting dependent variable from train and test")
            X_train,y_train,X_test,y_test = (
                train_array[:,:-1],
                train_array[:,-1],
                train_array[:,:-1],
                train_array[:,-1]
            )
            ##Train multiple model
            models = {
                'LinearRegression':LinearRegression(),
                'Lasso':Lasso(),
                'Ridge':Ridge(),
                'ElasticNet':ElasticNet()
            }
            mode_report:dict = evaluate_model(X_train,y_train,X_test,y_test,models)
            print(mode_report)
            print("\n=====================================================")
            logging.info(f'Model Report:{mode_report}')

            #to get the best model score from dictonary
            best_model_score = max(sorted(mode_report.values()))

            best_model_name = list(mode_report.keys())[
                list(mode_report.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]

            print(f'Best Model Found , Model Name : {best_model_name} , R2 Score : {best_model_score}')
            print('\n====================================================================================\n')
            logging.info(f'Best Model Found , Model Name : {best_model_name} , R2 Score : {best_model_score}')

            save_object(
                 file_path=self.model_trainer_config.trained_model_file_path,
                 obj=best_model
            )

        except Exception as e:
            logging.info("Exception occure at model training")
            raise CustomException(e,sys)
