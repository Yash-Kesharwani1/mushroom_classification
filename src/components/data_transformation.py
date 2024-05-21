import os
import sys
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
# from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
# from sklearn.preprocessing import TargetEncoder
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object
# from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import FunctionTransformer
import category_encoders as ce



class DataTransformationConfig:
    def __init__(self):
        self.preprocessor_obj_file_path = os.path.join('artifacts', 'preprocessor.pkl')

def count_encode(X):
    count_encoder = ce.CountEncoder()
    return count_encoder.fit_transform(X)


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()
        self.target_encoder = ce.CountEncoder()
        self.scaler = StandardScaler()

    # def list2int(list):
    #     return int(list[0]), int(list[1])
    
    # def get_data_transformer_object_for_target_variable(self):


    def get_data_transformer_object(self):
        '''
        This function is responsible for data transformation
        '''
        try:
            """
            yearUsed : 0
            abtest : 1
            vehicleType = 2
            gearbox = 3
            powerPS = 4
            model = 5
            kilometer = 6
            fuelType = 7
            brand = 8
            notRepairedDamage = 9
            """
            
            categorical_columns = [0,1,2,3,4,5,6,7]

            numerical_columns = [0, 1, 2, 3, 4, 5, 6, 7]

            cat_pipeline=Pipeline(

                steps=[
                ("CountEncoding",ce.CountEncoder())
                ]
            )

            num_pipeline= Pipeline(
                steps=[
                ("scaler",StandardScaler())
                ]
            )

            preprocessing_one = ColumnTransformer(
                [
                    ('cap-shape', 'drop', [0]),
                    ('cap-surface', 'drop', [1]),
                    ('cap-color', 'drop', [2]),
                    ('bruises', 'drop', [3]),
                    ('gill-attachment', 'drop', [5]),
                    ('gill-spacing', 'drop', [6]),
                    ('stalk-shape', 'drop', [9]),
                    ('stalk-root', 'drop', [10]),
                    ('veil-type', 'drop', [15]),
                    ('veil-color', 'drop', [16]),
                    ('ring-number', 'drop', [17]),
                    ('ring-type', 'drop', [18]),
                    ('population', 'drop', [20]),
                    ('habitat', 'drop', [21]),
                ], remainder='passthrough'
            )

            preprocessing_two = ColumnTransformer(
                [
                    ('cat_pipelines', cat_pipeline,categorical_columns)
                ], remainder='passthrough'
            )

            preprocessing_three = ColumnTransformer(
                [
                    ("num_pipeline",num_pipeline,numerical_columns)

                ],remainder='passthrough'
            )

            preprocessor = Pipeline(
                [
                    ('preprocessing_one', preprocessing_one),
                    ('preprocessing_two', preprocessing_two),
                    ('preprocessing_three', preprocessing_three)
                ]
            )

            
            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)

        
    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            preprocessing_obj = self.get_data_transformer_object()

            target_column_name = "class"

            input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)
            target_feature_train_df = train_df[target_column_name]
            # logging.info('Columns name after performing : ',input_feature_train_df.columns)

            # # printing the info of target_feature_train_df
            # print("printing the info of target_feature_train_df")
            # print(target_feature_train_df.info())

            input_feature_test_df = test_df
            target_feature_test_df = test_df[target_column_name]

            logging.info(f"Applying preprocessing object on training dataframe and testing dataframe.")
            

            # print(input_feature_train_df.info())

            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df, target_feature_train_df)

            print(input_feature_train_arr)

            # This print the name of features
            logging.info("Preprocessing one is done.")
            # print(preprocessing_obj.get_feature_names_out())
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)


            # Preprocessing target variables for train
            target_feature_train_df = pd.get_dummies(target_feature_train_df, drop_first=True)


            # Preprocessing target variables for test
            target_feature_test_df = pd.get_dummies(target_feature_test_df, drop_first=True)

            logging.info('Object_transformation is done.')

            train_arr = np.c_[
                input_feature_train_arr, np.array(target_feature_train_df)
            ]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            logging.info(f"Saved preprocessing object.")

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )
            train_arr_df = pd.DataFrame(train_arr)
            test_arr_df = pd.DataFrame(test_arr)
            train_arr_df.to_csv('notebook/train_arr_csv.csv')
            test_arr_df.to_csv('notebook/test_arr_csv.csv')


            return (
                train_arr, 
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )

        except Exception as e:
            raise CustomException(e, sys)