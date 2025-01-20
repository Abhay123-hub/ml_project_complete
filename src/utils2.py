import os
import sys
import numpy as np
import pandas as pd
import dill
import pickle
from src.exception import CustomException
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV


def model_evaluate(x_train,y_train,x_test,y_test,models,params):
    report_dict = {}
    try:
        for i in range(len(list(models.keys()))): ## iterating through ecah model and also to its hyperparameters
            model = list(models.values())[i] ## got the model
            para = params[list(models.keys())[i]] ## got the hyperparameters of the model
            gs = GridSearchCV(model,para,cv=3)
            gs.fit(x_train,y_train)
            model.set_params(**gs.best_params_)
            model.fit(x_train,y_train) ## training of the model on the provided dataset
            y_train_pred = model.predict(x_train)
            y_test_pred = model.predict(x_test)
            
            r2_score_train = r2_score(y_train,y_train_pred)
            r2_score_test = r2_score(y_test,y_test_pred)
            ## now i have r2_score of this model
            ## now i will add r2_score on the report dioctionary
            report_dict[list(models.keys())[i]] = r2_score_test
        return report_dict
    except Exception as e:
        raise CustomException(e,sys)



