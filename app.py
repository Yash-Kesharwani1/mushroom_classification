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
    app.run(host = '0.0.0.0', debug = True)

# from flask import Flask, request, render_template
# import math
# import os
# from src.pipeline.predict_pipeline import PredictPipeline
# from src.pipeline.predict_pipeline import CustomData
# import logging

# application = Flask(__name__)
# app = application

# PREPROCESSOR_PATH = os.path.join("artifacts", "preprocessor.pkl")

# # Route for the home page
# @app.route("/")
# def index():
#     return render_template('index.html')

# @app.route('/predict_datapoint', methods=['GET','POST'])
# def predict_datapoint():
#     if request.method == 'GET':
#         return render_template('home.html')
#     else:
#         # Retrieve form data
#         order = request.form.get('order')
#         gill_size = request.form.get('gill_size')
#         gill_color = request.form.get('gill_color')
#         stalk_surface_above_ring = request.form.get('stalk_surface_above_ring')
#         stalk_surface_below_ring = request.form.get('stalk_surface_below_ring')
#         stalk_color_above_ring = request.form.get('stalk_color_above_ring')
#         stalk_color_below_ring = request.form.get('stalk_color_below_ring')
#         spore_print_color = request.form.get('spore_print_color')

#         # Debugging: Print received form data
#         logging.info(f"Received data: order={order}, gill_size={gill_size}, gill_color={gill_color}, stalk_surface_above_ring={stalk_surface_above_ring}, stalk_surface_below_ring={stalk_surface_below_ring}, stalk_color_above_ring={stalk_color_above_ring}, stalk_color_below_ring={stalk_color_below_ring}, spore_print_color={spore_print_color}")

#         # Ensure all fields have values
#         if None in [order, gill_size, gill_color, stalk_surface_above_ring, stalk_surface_below_ring, stalk_color_above_ring, stalk_color_below_ring, spore_print_color]:
#             return render_template('home.html', results="Incomplete form data.")

#         # Create a CustomData instance
#         data = CustomData(
#             order=order,
#             gill_size=gill_size,
#             gill_color=gill_color,
#             stalk_surface_above_ring=stalk_surface_above_ring,
#             stalk_surface_below_ring=stalk_surface_below_ring,
#             stalk_color_above_ring=stalk_color_above_ring,
#             stalk_color_below_ring=stalk_color_below_ring,
#             spore_print_color=spore_print_color
#         )

#         # Convert to DataFrame
#         pred_df = data.get_data_as_data_frame()
#         logging.info(f"Prediction DataFrame: {pred_df}")

#         # Use the prediction pipeline
#         try:
#             predict_pipeline = PredictPipeline()
#             results = predict_pipeline.predict(pred_df)
#             prediction = math.floor(results[0])
#         except Exception as e:
#             logging.error(f"Error during prediction: {e}")
#             return render_template('home.html', results="Error during prediction")

#         return render_template('home.html', results=prediction)

# if __name__ == '__main__':
#     app.run(host='0.0.0.0', debug=True)




# from flask import Flask, request, render_template
# import math
# import os
# from src.pipeline.predict_pipeline import PredictPipeline
# from src.pipeline.predict_pipeline import CustomData
# import logging
# import pandas as pd

# application = Flask(__name__)
# app = application

# PREPROCESSOR_PATH = os.path.join("artifacts", "preprocessor.pkl")

# # Route for the home page
# @app.route("/")
# def index():
#     return render_template('index.html')

# @app.route('/predict_datapoint', methods=['GET', 'POST'])
# def predict_datapoint():
#     if request.method == 'GET':
#         return render_template('home.html')
#     else:
#         # Retrieve form data
#         order = request.form.get('order')
#         gill_size = request.form.get('gill_size')
#         gill_color = request.form.get('gill_color')
#         stalk_surface_above_ring = request.form.get('stalk_surface_above_ring')
#         stalk_surface_below_ring = request.form.get('stalk_surface_below_ring')
#         stalk_color_above_ring = request.form.get('stalk_color_above_ring')
#         stalk_color_below_ring = request.form.get('stalk_color_below_ring')
#         spore_print_color = request.form.get('spore_print_color')

#         # Debugging: Print received form data
#         logging.info(f"Received data: order={order}, gill_size={gill_size}, gill_color={gill_color}, stalk_surface_above_ring={stalk_surface_above_ring}, stalk_surface_below_ring={stalk_surface_below_ring}, stalk_color_above_ring={stalk_color_above_ring}, stalk_color_below_ring={stalk_color_below_ring}, spore_print_color={spore_print_color}")

#         # Ensure all fields have values
#         if None in [order, gill_size, gill_color, stalk_surface_above_ring, stalk_surface_below_ring, stalk_color_above_ring, stalk_color_below_ring, spore_print_color]:
#             return render_template('home.html', results="Incomplete form data.")

#         # Create a CustomData instance
#         data = CustomData(
#             order=order,
#             gill_size=gill_size,
#             gill_color=gill_color,
#             stalk_surface_above_ring=stalk_surface_above_ring,
#             stalk_surface_below_ring=stalk_surface_below_ring,
#             stalk_color_above_ring=stalk_color_above_ring,
#             stalk_color_below_ring=stalk_color_below_ring,
#             spore_print_color=spore_print_color
#         )

#         # Convert to DataFrame
#         pred_df = data.get_data_as_data_frame()
#         logging.info(f"Prediction DataFrame: {pred_df}")

#         # Ensure the DataFrame has the correct shape
#         if pred_df.shape[0] == 1:
#             pred_df = pred_df.values.reshape(1, -1)
#         logging.info(f"Reshaped DataFrame: {pred_df}")

#         # Use the prediction pipeline
#         try:
#             predict_pipeline = PredictPipeline()
#             results = predict_pipeline.predict(pred_df)
#             prediction = math.floor(results[0])
#         except Exception as e:
#             logging.error(f"Error during prediction: {e}")
#             return render_template('home.html', results="Error during prediction")

#         return render_template('home.html', results=prediction)

# if __name__ == '__main__':
#     app.run(host='0.0.0.0', debug=True)
