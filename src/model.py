import os.path
import shutil
import mlflow
# from mlflow.tracking.artifact_utils import
from mlflow.models import infer_signature
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import mean_squared_error, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from typing import List, Dict, Union, Optional
import logging


class ColumnRemapper(BaseEstimator, TransformerMixin):
    """
    Transformer to remap values in specified columns.
    """

    def __init__(self, column_mapping: Dict[str, Dict[str, str]]):
        self.column_mapping = column_mapping

    def fit(self, data, outcomes=None):
        return self

    def transform(self, data):
        # TODO: add option to do in place to save memory for large datasets
        data = data.copy()

        for col, remap in self.column_mapping.items():
            data[col] = data[col].map(remap)

        return data

class MLflowModel(BaseEstimator):
    """
    A wrapper class for ML models with MLFlow tracking.
    Includes feature engineering, preprocessing, training, and prediction.
    """

    def __init__(
            self,
            model,
            experiment_name: str,
            numeric_features: List[str],
            categorical_features: List[str],
            target_col: str,
            problem_type: str = 'regression',
            feature_engineering_steps: Optional[List[tuple]] = None,
            column_mapping: Optional[Dict[str, Dict[str, str]]] = None,
            tracking_uri: Optional[str] = None
    ):
        """
        Initialize the model wrapper.

        Args:
            model: Any sklearn-compatible model
            experiment_name: Name for MLflow experiment
            numeric_features: List of numerical column names
            categorical_features: List of categorical column names
            target_col: Target column name
            problem_type: 'regression' or 'classification'
            feature_engineering_steps: List of (name, transformer) tuples for feature engineering
            column_mapping: Dictionary of column-specific remapping rules
            tracking_uri: MLflow tracking URI
        """
        self.model = model
        self.experiment_name = experiment_name
        self.numeric_features = numeric_features
        self.categorical_features = categorical_features
        self.target_col = target_col
        self.problem_type = problem_type
        self.feature_engineering_steps = feature_engineering_steps or []
        self.column_mapping = column_mapping or {}

        self.pipeline = self._create_pipeline()

        # Populated during fit
        self.signature = None
        # TODO: hacky way of doing model version, but works for local model saving for now
        # self.model_version = 1

        # Set up MLFlow
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)

        # Get or create experiment
        try:
            self.experiment_id = mlflow.create_experiment(experiment_name)
        except:
            self.experiment_id = mlflow.get_experiment_by_name(experiment_name).experiment_id

        # Initialize logger
        self.logger = logging.getLogger(__name__)

    def set_model(self, model) -> None:
        self.model = model
        self.pipeline = self._create_pipeline()

    def _create_pipeline(self) -> Pipeline:
        """Create the preprocessing and modeling pipeline"""

        # scale numeric columns
        numeric_transformer = Pipeline(
            steps=[
                # ('imputer', SimpleImputer(strategy='mean')),
                ('scaler', StandardScaler())
            ]
        )

        # fill missing values with 'missing and one hot encode categorical columns
        categorical_transformer = Pipeline(
            steps=[
                ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
                ('onehot', OneHotEncoder(handle_unknown='ignore'))
            ]
        )

        preprocesser = ColumnTransformer(
            transformers=[
                ('numeric', numeric_transformer, self.numeric_features),
                ('categorical', categorical_transformer, self.categorical_features)
            ],
            # drop any columns not specified in numeric_features or categorical_features
            remainder='drop'
        )

        pipeline = Pipeline(
            steps=[
                ('preprocessor', preprocesser),
                ('model', self.model)
            ]
        )

        # TODO: Add feature engineering steps and column remapping
        # # Add column remapping
        # steps.append(('remapping', ColumnRemapper(self.column_mapping)))
        #
        # # Add feature engineering steps
        # steps.extend(self.feature_engineering_steps)

        return pipeline

    def _log_parameters(self, params: Dict):
        """Log parameters to MLFlow"""
        for key, value in params.items():
            mlflow.log_param(key, value)

    def _log_metrics(self, metrics: Dict):
        """Log metrics to MLFlow"""
        for key, value in metrics.items():
            mlflow.log_metric(key, value)

    def evaluate(self, y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
        """Calculate evaluation metrics based on problem type"""
        metrics = {}

        if self.problem_type == 'regression':
            metrics['mse'] = mean_squared_error(y_true, y_pred)
            metrics['rmse'] = np.sqrt(metrics['mse'])
            metrics['mae'] = np.mean(np.abs(y_true - y_pred))
        elif self.problem_type == 'classification':
            metrics['accuracy'] = accuracy_score(y_true, y_pred)
            metrics['precision'] = precision_score(y_true, y_pred, average='weighted')
            metrics['recall'] = recall_score(y_true, y_pred, average='weighted')
            metrics['f1'] = f1_score(y_true, y_pred, average='weighted')
            metrics['auc_score'] = roc_auc_score(y_true, y_pred, average='weighted')
        else:
            raise ValueError(f"Unrecognized problem type: {self.problem_type}")

        return metrics

    def fit(self, data: pd.DataFrame, outcome_data: pd.Series | np.ndarray, **kwargs):
        """
        Fit the model and log to MLflow

        Args:
            data: Feature DataFrame
            outcome_data: Target variable
            **kwargs: Additional arguments passed to model.fit()
        """
        # mlflow.autolog()

        # NOTE: we don't need to call `start_run` because Azure ML already does that
        # with mlflow.start_run(experiment_id=self.experiment_id):
        try:
            # fit the pipeline create the model signature
            self.pipeline.fit(data, outcome_data)
            self.signature = infer_signature(data, self.pipeline.predict(data))

            # Log parameters
            params = {
                'model_type': str(type(self.model).__name__),
                'numeric_features': self.numeric_features,
                'categorical_features': self.categorical_features,
                'feature_engineering_steps': str(self.feature_engineering_steps)
            }
            self._log_parameters(params)

            # Log model
            mlflow.sklearn.log_model(self.pipeline, self.experiment_name, signature=self.signature)

            self.logger.info("Model training completed successfully")

        except Exception as e:
            self.logger.error(f"Error during model training: {str(e)}")
            raise

        return self

    def predict(self, data: pd.DataFrame) -> np.ndarray:
        """
        Make predictions on new data

        Args:
            data: Feature DataFrame

        Returns:
            np.ndarray: Predictions
        """
        try:
            predictions = self.pipeline.predict(data)
            return predictions
        except Exception as e:
            self.logger.error(f"Error during prediction: {str(e)}")
            raise

    # save model
    def save_model(self, path: str) -> None:
        """
        Save the model to a file

        Args:
           path: File path
        """
        # delete model if it already exists
        # TODO: create a new version instead of deleting it
        if os.path.exists(path):
            # delete directory and all its contents
            shutil.rmtree(path)

        mlflow.sklearn.save_model(self.pipeline, path, signature=self.signature)

        self.logger.info(f"Model saved to {path}")
        # TODO: check if the model needs this python file to save the model properly

    # load model
    def load_model(self, path: str) -> None:
        """
        Load the model from a file

        Args:
            path: File path
        """
        self.pipeline = mlflow.sklearn.load_model(path)

    def evaluate_and_log(self, data: pd.DataFrame, outcome_data: pd.Series | np.ndarray):
        """
        Evaluate model performance and log metrics to MLflow

        Args:
            data: Feature DataFrame
            outcome_data: True target values
        """
        # mlflow.autolog()

        # NOTE: we don't need to call `start_run` because Azure ML already does that
        # with mlflow.start_run(experiment_id=self.experiment_id):
        try:
            predictions = self.predict(data)
            metrics = self.evaluate(outcome_data, predictions)
            self._log_metrics(metrics)
            self.logger.info("Evaluation metrics logged successfully")
            return metrics
        except Exception as e:
            self.logger.error(f"Error during evaluation: {str(e)}")
            raise