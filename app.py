from flask import Flask,render_template,request
import numpy as np
import pandas as pd
from src.exception import CustomException
import sys

from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData,PredictionPipeLine
from src.logger import logging
application = Flask(__name__)
app = application

## route for the home page
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predictdata',methods = ["GET","POST"])
def predict_datapoint():
    if request.method=="GET":
        return render_template('home.html')
    else:
        try:
                data = CustomData(
                    gender = request.form.get('gender'),
                    race_ethnicity = request.form.get("ethnicity"),
                    parental_level_of_education=request.form.get("parental_level_of_education"),
                    lunch = request.form.get("lunch"),
                    test_preparation_course=request.form.get("test_preparation_course"),
                    writing_score= float(request.form.get('writing_score')),
                    reading_score = float(request.form.get('reading_score')),
                  
                    

                )
                df = data.data_to_pandas()
                logging.info("converted user input into pandas data frame")
                print(df)
                predict_pipeline = PredictionPipeLine()
                logging.info("doing prediction")
                result = predict_pipeline.predict(df)
                logging.info("prediction is done")

                return render_template('home.html',results=result[0])
        except Exception as e:
             raise CustomException(e,sys)
   

if __name__ == "__main__":
    app.run(host='0.0.0.0')

