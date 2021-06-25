import os
import pandas as pd
import numpy as np

from utils import processing_data_from_excel, predict_json

battery_x_test = pd.read_excel(
    r"data/CX2_36_input_data.xlsx", sheet_name="Sheet1")
X_battery_test = battery_x_test.to_numpy()
X_battery_test = X_battery_test[:, 1:]

data = processing_data_from_excel(X_battery_test).tolist()
print(data)
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "capstone-311510-507632fc6686.json"
REGION = "us-central1"
PROJECT = "capstone-311510"
MODEL = "battery_01"
predict_json(project=PROJECT, region=REGION, model=MODEL, instances=data)
