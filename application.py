import os
from flask import Flask, request, render_template
import math
from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import PredictPipeline
from src.pipeline.predict_pipeline import CustomData
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_building import ModelTrainer
from src.utils import load_object
import logging
import numpy as np
from src.exception import CustomException

application = Flask(__name__)
app = application

PREPROCESSOR_PATH = os.path.join("artifacts", "preprocessor.pkl")

# Route for a home page
@app.route("/")
def index():
    return render_template('index.html')



@app.route('/predict_datapoint', methods=['GET','POST'])
def predict_datapoint():
        
    if request.method == 'GET':
        return render_template('home.html')
    else:
        data = CustomData(
            odor = request.form.get('odor'),
            gill_size = request.form.get('gill_size'),
            gill_color = request.form.get('gill_color'),
            stalk_surface_above_ring = request.form.get('stalk_surface_above_ring'),
            stalk_surface_below_ring = request.form.get('stalk_surface_below_ring'),
            stalk_color_above_ring = request.form.get('stalk_color_above_ring'),
            stalk_color_below_ring = request.form.get('stalk_color_below_ring'),
            spore_print_color = request.form.get('spore_print_color')
        )

        pred_df = data.get_data_as_data_frame()
        print(pred_df)

        predict_pipeline = PredictPipeline()
        results = predict_pipeline.predict(pred_df)
        return render_template('home.html', results = math.floor(results[0]))


if __name__ == '__main__':
    app.run(host = '0.0.0.0')