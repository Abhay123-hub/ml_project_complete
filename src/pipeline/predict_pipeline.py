import sys
import os
import pandas as pd
from src.exception import CustomException
from src.logger import logging

from src.utils import load_object

class CustomData:
    def __init__(self,
            gender:str,
            race_ethnicity:str,
            parental_level_of_education:str,
            lunch:str,
            test_preparation_course:str,
            reading_score:int,
            writing_score:int,
    ):
        self.gender = gender
        self.race_ethnicity=race_ethnicity
        self.parental_level_of_education=parental_level_of_education
        self.lunch=lunch
        self.test_prepration_course=test_preparation_course
        self.reading_score=reading_score
        self.writing_score=writing_score
    def data_to_pandas(self):
     try:
        data_dict = {
           "gender":[self.gender],
           "race_ethnicity":[self.race_ethnicity],
           "parental_level_of_education":[self.parental_level_of_education],
           "lunch":[self.lunch],
           "test_preparation_course":[self.test_prepration_course],
            "writing_score":[self.writing_score],
           "reading_score":[self.reading_score],
          
        }
        return pd.DataFrame(data_dict)
     except Exception as e:
        raise CustomException(e,sys)
     

## the above code was for converting the user provided data into pandas dataframe
## because my machine learning model will be able to work on pandas dataframe
## now i will be writting the code for Prediction PipeLine class
## inside this class i will be loading my preprocessor, i will transform the data
## then i will be loading the model
## and i will do prediction using my model


class PredictionPipeLine:
   def __init__(self):
      pass
   def predict(self,features):
      try:
         model_path = os.path.join("artifacts","model.pkl")
         processor_path = os.path.join("artifacts","processor.pkl")
         model = load_object(path=model_path)
         processor = load_object(path=processor_path)
         processed_data = processor.transform(features)
         pred = model.predict(processed_data)
         return pred
        
        

      except Exception as e:
         raise CustomException(e,sys)
      

