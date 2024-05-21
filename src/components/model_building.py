import os
import sys
from dataclasses import dataclass

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import r2_score

from src.exception import CustomException
from src.logger import logging

from src.utils import save_object, evaluate_models, load_object

@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join("artifacts","model.pkl")


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()


    def initiate_model_trainer(self,train_array, test_array):
        try:
            logging.info("Split training and test input data")
            X_train, y_train = train_array[:, 0:8], train_array[:, 8]
            X_test, y_test = test_array[:, 0:8], test_array[:, 8]

            print('This is y_train : ',y_train)
            print('This is y_test : ',y_test)
            print('Size of x_train and x_test : ',(X_train.shape),', ',(X_test.shape))
            models = {
                # "Random Forest": RandomForestRegressor(),
                # "Decision Tree": DecisionTreeRegressor()
                # "Gradient Boosting": GradientBoostingRegressor(),
                "Logistic Regression": LogisticRegression()
                # "XGBRegressor": XGBRegressor()
                # "CatBoosting Regressor": CatBoostRegressor(verbose=False),
                # "AdaBoost Regressor": AdaBoostRegressor()
            }

            params = {
                # "Decision Tree": {
                #     'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                #     'splitter':['best','random'],
                #     'max_features':['sqrt','log2'],
                # }
                # "Random Forest":{
                #     # 'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                 
                #     # 'max_features':['sqrt','log2',None],
                #     'n_estimators': [4,128,256,1024]
                # },
                # "Gradient Boosting":{
                #     # 'loss':['squared_error', 'huber', 'absolute_error', 'quantile'],
                #     'learning_rate':[0.01],
                #     # 'subsample':[0.6,0.7,0.75,0.8,0.85,0.9],
                #     # 'criterion':['squared_error', 'friedman_mse'],
                #     # 'max_features':['auto','sqrt','log2'],
                #     'n_estimators': [1024]
                # }
                "Logistic Regression":{}
                # "XGBRegressor":{
                #     'learning_rate':[.1,.01,.05,.001],
                #     'n_estimators': [8,16,32,64,128,256,1024]
                # }
                # "CatBoosting Regressor":{
                #     'depth': [6,8,10],
                #     'learning_rate': [0.01, 0.05, 0.1],
                #     'iterations': [30, 50, 100]
                # },
                # "AdaBoost Regressor":{
                #     'learning_rate':[.1,.01,0.5,.001],
                #     # 'loss':['linear','square','exponential'],
                #     'n_estimators': [8,16,32,64,128,256,1024]
                # }
            }

            model_report = evaluate_models(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,models=models, param=params)

            

            # Get best model from dict
            best_model_name = max(model_report, key=model_report.get)
            best_model_score = model_report[best_model_name]

            logging.info(f'Best model: {best_model_name}, Score: {best_model_score}')

            best_model = models[best_model_name]

            if best_model_score < 0.6:
                logging.info("Custom Exception is raised.")
                raise CustomException("No best model found with R^2 score less than 0.6")


            logging.info(f"Best model found: {best_model_name} with score {best_model_score}")

            # Fit best model
            best_model.fit(X_train, y_train)

            # Save trained model
            save_object(file_path=ModelTrainerConfig.trained_model_file_path, obj=best_model)

            # Predict on test set
            y_pred = best_model.predict(X_test)

            # Calculate R^2 score
            r2_score_value = r2_score(y_test, y_pred)

            logging.info(f"R^2 Score on test set: {r2_score_value}")

            return r2_score_value

        except Exception as e:
            raise CustomException(e, sys)