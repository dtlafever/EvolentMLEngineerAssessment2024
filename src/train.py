from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from model import MLflowModel
import mltable
import argparse

# # Create custom feature engineering steps if needed
# class CustomTransformer(BaseEstimator, TransformerMixin):
#     def fit(self, X, y=None):
#         return self
#
#     def transform(self, X):
#         # Add your custom transformations here
#         return X

def remap_target_column(data, target_col):
    data[target_col] = data[target_col].map({"Yes": 1, "No": 0})
    return data


# MAIN

# parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("--input", help="mltable to read")
args = parser.parse_args()

target_col = "Readmitted"

drop_cols = [
    "Patient_ID"
]

categorical_cols = [
    "Gender",
    "Admission_Type",
    "Diagnosis",
    "A1C_Result"
]

numerical_cols = [
    "Age",
    "Num_Lab_Procedures",
    "Num_Medications",
    "Num_Outpatient_Visits",
    "Num_Inpatient_Visits",
    "Num_Emergency_Visits",
    "Num_Diagnoses"
]

# feature_engineering = [
#     ('custom_transform', CustomTransformer())
# ]

model_name = 'healthcare-model'

# Initialize the model wrapper
model = MLflowModel(
    model=RandomForestClassifier(),
    experiment_name=model_name,
    numeric_features=numerical_cols,
    categorical_features=categorical_cols,
    target_col=target_col,
    problem_type='classification',
    feature_engineering_steps=None,
    column_mapping=None
)

# load mltable
tbl = mltable.load(args.input)

# load data
data = tbl.to_pandas_dataframe()

# split to train and test
train_data, test_data, train_outcome, test_outcome = train_test_split(
    data.drop(target_col, axis=1), data[target_col], test_size=0.2, random_state=42)

# Train the model
model.fit(train_data, train_outcome)

# # Make predictions
# predictions = model.predict(test_data)

# Evaluate and log metrics
metrics = model.evaluate_and_log(test_data, test_outcome)

# save model
# model.save_model(model.experiment_name)
# mlflow.register_model()
