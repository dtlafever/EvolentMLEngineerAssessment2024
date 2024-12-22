import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from src.model import MLflowModel
import mlflow
from xgboost import XGBClassifier

def main():
    # path to dataset
    data_path = './data/hospital_readmissions_clean.parquet'

    # load dataset
    data = pd.read_parquet(data_path)

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

    model_name = 'healthcare-model'
    model_path = f"./models/{model_name}"

    models_to_test = {
        'RandomForest': RandomForestClassifier(random_state=42),
        'GradientBoosting': GradientBoostingClassifier(random_state=42),
        'LogisticRegression': LogisticRegression(random_state=42, solver='liblinear'),
        "XGBoost": XGBClassifier(random_state=42, eval_metric="logloss")
    }

    # Initialize the model wrapper
    model_wrapper = MLflowModel(
        model=RandomForestClassifier(),
        experiment_name=model_name,
        numeric_features=numerical_cols,
        categorical_features=categorical_cols,
        target_col=target_col,
        problem_type='classification',
        feature_engineering_steps=None,
        column_mapping=None
    )

    # split to train and test
    train_data, test_data, train_outcome, test_outcome = train_test_split(
        data.drop(target_col, axis=1), data[target_col], test_size=0.1, random_state=42)

    train_outcome = train_outcome.apply(lambda x: 1 if x == "Yes" else 0)
    test_outcome = test_outcome.apply(lambda x: 1 if x == "Yes" else 0)

    for model_type, model in models_to_test.items():
        print(f"Training {model_type} model...")
        run_name = f"{model_type}-{model_name}"
        run_name_path = f"./models/{run_name}"
        with mlflow.start_run(run_name=run_name):
            model_wrapper.set_model(model)
            # Train the model
            model_wrapper.fit(train_data, train_outcome)

            # Evaluate and log metrics
            metrics = model_wrapper.evaluate_and_log(test_data, test_outcome)
            for metric_name, metric_value in metrics.items():
                print(f"\t{metric_name}:{metric_value:.4f}")

            # save model
            model_wrapper.save_model(run_name_path)
            print(f"Model saved to {run_name_path}")




if __name__ == "__main__":
    main()