from src.pipeline.predict_pipeline import CustomData,PredictionPipeLine
from src.exception import CustomException
import pandas as pd
from src.utils1 import load_object
import sys
model = load_object("artifacts\model.pkl")
data = pd.read_csv(r"artifacts\test.csv")
data = data.head()
pred = PredictionPipeLine()
result = pred.predict(data)
print(result)