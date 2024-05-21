import sys, os
import pandas as pd
from src.exception import CustomException
from src.utils import load_object, save_object
from src.logger import logging


class PredictPipeline:
    def __init__(self) -> None:
        pass

    # def predict(self, features):
    #     try:
    #         # print('The shape of the features is : ',features.shape)
    #         model_path = 'artifacts\model.pkl'
    #         preprocessor_path = 'artifacts\preprocessor.pkl'
    #         model = load_object(file_path=model_path)
    #         preprocessor = load_object(file_path=preprocessor_path)
    #         data_scaled = preprocessor.transform(features)
    #         predict = model.predict(data_scaled)
    #         return predict
    #     except Exception as e:
    #         raise CustomException(e,sys)
    
    def fit_preprocessor(self, X_train):
        try:
            preprocessor_path = os.path.join('artifacts', 'proprocessor.pkl')
            preprocessor = load_object(file_path=preprocessor_path)
            preprocessor.fit(X_train)  # Fit the preprocessor to training data
            save_object(preprocessor_path, preprocessor)  # Save the fitted preprocessor
        except Exception as e:
            raise CustomException(e, sys)
        
   # This function will predict the target variable
    def predict(self, features):
        try:
            model_path = os.path.join("artifacts", "model.pkl")
            preprocessor_path = os.path.join('artifacts', 'preprocessor.pkl')

            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)

            # print the recived features
            print(f"Recived Features : {features}")

            # Transform the input features
            data_scaled = preprocessor.transform(features)

            print("data_scaled/pred_df : ",data_scaled)
            logging.info("data_scaled/pred_df : {data_scaled}")

            # Predict using the model
            preds = model.predict(data_scaled)
            return preds
        
        except Exception as e:
            raise CustomException(e, sys)
        
# This class map the data given from the form to the backend input features and helps for the predictions


class CustomData:
    def __init__(
        self,
        odor: str,
        gill_size: str,
        gill_color: str,
        stalk_surface_above_ring: str,
        stalk_surface_below_ring: str,
        stalk_color_above_ring: str,
        stalk_color_below_ring: str,
        spore_print_color: str
    ):

        self.odor = odor

        self.gill_size = gill_size

        self.gill_color = gill_color

        self.stalk_surface_above_ring = stalk_surface_above_ring

        self.stalk_surface_below_ring = stalk_surface_below_ring

        self.stalk_color_above_ring = stalk_color_above_ring

        self.stalk_color_below_ring = stalk_color_below_ring

        self.spore_print_color = spore_print_color

    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                'odor': [self.odor],
                'gill_size': [self.gill_size],
                'gill_color': [self.gill_color],
                'stalk_surface_above_ring': [self.stalk_surface_above_ring],
                'stalk_surface_below_ring': [self.stalk_surface_below_ring],
                'stalk_color_above_ring': [self.stalk_color_above_ring],
                'stalk_color_below_ring': [self.stalk_color_below_ring],
                'spore_print_color': [self.spore_print_color]
            }

            # Create a DataFrame with the same columns as used during training
            columns = ['odor', 'gill_size', 'gill_color', 'stalk_surface_above_ring', 'stalk_surface_below_ring', 'stalk_color_above_ring', 'stalk_color_below_ring', 'spore_print_color']
            df = pd.DataFrame(custom_data_input_dict, columns=columns)

            return df

        except Exception as e:
            raise CustomException(e, sys)
