# standard libs
import time
from pathlib import Path

# Azure ML Connection and Data assets
from azure.ai.ml import MLClient, command, Input
from azure.identity import DefaultAzureCredential
from azure.ai.ml.entities import Data, AmlCompute, Environment
from azure.ai.ml.constants import AssetTypes
from mltable import DataType, from_parquet_files
from azure.ai.ml.exceptions import ValidationException
from azure.core.exceptions import ResourceNotFoundError

# from azureml.core import Workspace
from azureml.core.compute import ComputeTarget#, AmlCompute
from azureml.core.compute_target import ComputeTargetException

# misc libs
from decouple import config

def get_or_create_cpu_compute(ml_client: MLClient,
                              compute_name: str,
                              vm_size: str = "Standard_DS3_v2",
                              location: str = "eastus2",
                              low_priority: bool = False,
                              min_nodes: int = 0,
                              max_nodes: int = 4) -> ComputeTarget:
    """
    Gets the cpu compute target if it exists, otherwise creates a new one.

    :param ml_client: a valid MLClient object for connecting to Azure ML Studio
    :param compute_name: the name of the compute target to create or get
    :param vm_size: defaults to "Standard_DS3_v2", which is good for small dataset training (4core, 14GB RAM, 28GB storage, $0.23/hr).
    :param location: defaults to "eastus2"
    :param low_priority: defaults to False
    :param min_nodes: defaults to 0
    :param max_nodes: defaults to 4
    :return: the created compute target object
    """

    try:
        compute_target = ml_client.compute.get(compute_name)
        print(f"Found existing compute target: {compute_name}")
    except ResourceNotFoundError:
        print(f"Creating new compute target: {compute_name}")

        if low_priority:
            tier = "LowPriority"
        else:
            tier = "Dedicated"

        compute_config = AmlCompute(
            name=compute_name,
            type="amlcompute",
            size=vm_size,
            location=location,
            min_instances=min_nodes,
            max_instances=max_nodes,
            tier=tier,
            idle_time_before_scale_down=120
        )
        compute_target = ml_client.begin_create_or_update(compute_config).result()

    return compute_target

def upload_and_get_data(ml_client: MLClient,
                         data_asset_name: str,
                         data_path: str | Path,
                         column_types: dict[str, DataType],
                         description: str = "") -> Data:
    """
    Uploads the data to Azure ML Studio and returns the data asset object.

    :param ml_client: a valid MLClient object for connecting to Azure ML Studio
    :param data_asset_name: the name of the data asset to create or get
    :param data_path: the path to the data file to upload (locally)
    :param column_types: a dictionary of column names and their respective data types
    :param description: a description of the data asset
    :return: the created data asset object
    """

    # List of dictionaries containing the path to the data file used for MLTable creation
    paths = [{"file": data_path}]

    # Create MLTable file to make the ml table in azure in the next step
    ml_table = from_parquet_files(paths)
    # TODO: hard coded path. Make this a parameter
    ml_table = ml_table.save("./data")
    # ml_table = ml_table.convert_column_types(column_types)

    # load data
    # data = pd.read_parquet(data_path)
    # data_df = ml_table.to_pandas_dataframe()
    data_version = time.strftime("%Y.%m.%d.%H%M%S", time.gmtime())

    # create the data asset object to be uploaded to azure
    data_asset = Data(
        path="./data",
        type=AssetTypes.MLTABLE,
        description=description,
        name=data_asset_name,
        version=data_version
    )

    # create the data asset in your workspace
    data_asset = ml_client.data.create_or_update(data_asset)

    return data_asset

def train_model(ml_client: MLClient,
                model_name: str,
                data_asset_id: str,
                compute_name: str,
                environment: Environment | str,
                script_path: str = "./src",
                hyperparameters: dict[str, any] = None):
    """
    Create Azure ML Studio job and train a MLFlow model. Once the model is trained, register it in Azure ML Studio.
    :param ml_client:
    :param model_name:
    :param data_asset_id:
    :param compute_name:
    :param script_path:
    :param environment:
    :param hyperparameters:
    :return:
    """

    # Create a job to run the training script
    job = command(
        code=script_path,
        command="python train.py --input ${{inputs.hosp_data}}",
        inputs={"hosp_data": Input(type="mltable", path=data_asset_id)},
        environment=environment,
        compute=compute_name,
        display_name=f"train-model-job-{model_name}",
        # experiment_name=f"train-model-{model_name}",
    )

    returned_job = ml_client.jobs.create_or_update(job)
    print(f"Job URL: {returned_job.studio_url}")

    # Register the model
    # NOTE: already done in azure ML studio
    # model = ml_client.models.create_or_update()

def main():
    # 1. Initialize Azure ML workspace
    print("Initializing Azure ML workspace...")
    # NOTE: DefaultAzureCredential() will require `az login` to be run in the terminal via azure cli
    ml_client = MLClient.from_config(credential=DefaultAzureCredential())
    print("Azure ML workspace initialized!")

    # 2. Get data and upload it to Azure ML
    print("Looking for data asset in Azure ML...")

    data_version = "1"
    data_asset_name = "hospital_readmissions"
    # path to dataset
    data_path = './data/hospital_readmissions_clean.parquet'
    # Set the data types for each column
    column_types = {
        "Patient_ID": DataType.to_string(),
        "Age": DataType.to_int(),
        "Gender": DataType.to_string(),
        "Admission_Type": DataType.to_string(),
        "Diagnosis": DataType.to_string(),
        "Num_Lab_Procedures": DataType.to_int(),
        "Num_Medications": DataType.to_int(),
        "Num_Outpatient_Visits": DataType.to_int(),
        "Num_Inpatient_Visits": DataType.to_int(),
        "Num_Emergency_Visits": DataType.to_int(),
        "Num_Diagnoses": DataType.to_int(),
        "A1C_Result": DataType.to_string(),
        "Readmitted": DataType.to_string()
    }
    data_description = "Hospital Readmissions Data"

    try:
        data_asset = ml_client.data.get(data_asset_name, version=data_version)
        print(f"Data asset '{data_asset_name}' found!")
    except ResourceNotFoundError:
        print(f"Data asset '{data_asset_name}' not found.")
        print("Uploading and Getting data to Azure ML...")
        data_asset = upload_and_get_data(ml_client,
                                         data_asset_name,
                                         data_path,
                                         column_types,
                                         data_description)
        print(f"Data asset '{data_asset_name}' uploaded/got successfully!")

    exit()

    # 3. Create or get compute target
    print("Creating/Getting compute target...")
    compute_name = "basic-4-node-cpu-cluster"

    compute_target = get_or_create_cpu_compute(ml_client, compute_name, low_priority=False)
    print(f"Compute target '{compute_name}' created/got successfully!")

    # 4. Train the model
    print("Training the model...")
    # NOTE: if you want to not create a dedicated environment and instead use your own docker image and conda file,
    #       uncomment this code.
    # environment = Environment(image="mcr.microsoft.com/azureml/curated/minimal-app-quickstart:11",
    #                           conda_file="./job-env/conda_dependencies.yml")
    # This environment was already created in Azure ML Studio.
    environment = "simple-ubuntu2204-py310-cpu-inference:latest"

    train_model(ml_client, "hospital_readmissions_model", data_asset.id, compute_name, environment)
    print("Job created. Please track it in Azure. Once complete and you are happy with the results, you can deploy the model using 'azure_deploy_model.py' script.")

if __name__ == "__main__":
    main()