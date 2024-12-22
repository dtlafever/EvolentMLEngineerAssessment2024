from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential
from azure.ai.ml.entities import ManagedOnlineEndpoint, ManagedOnlineDeployment

def create_online_endpoint(ml_client: MLClient,
                           endpoint_name: str,
                           model_name: str,
                           model_version: str,
                           instance_type: str = "Standard_DS3_v2",
                           instance_count: int = 1):
    """
        Create an online endpoint for a registered model in Azure ML Studio.

        Args:
            ml_client (str): Name of the Azure ML workspace
            endpoint_name (str): Name for the new endpoint
            model_name (str): Name of the registered model
            model_version (int): Version of the registered model
            instance_type (str): VM size for deployment (default: Standard_DS3_v2)
            instance_count (int): Number of instances (default: 1)
        """

    # Create a new endpoint
    endpoint = ManagedOnlineEndpoint(
        name=endpoint_name,
        description=f"Online endpoint for {model_name}",
        auth_mode="key",
    )

    ml_client.online_endpoints.begin_create_or_update(endpoint).result()

    # Get registered model
    model = ml_client.models.get(model_name, model_version)

    # Create Deployment
    deployment = ManagedOnlineDeployment(
        name=f"deployment-{endpoint_name}",
        endpoint_name=endpoint_name,
        model=model,
        instance_type=instance_type,
        instance_count=instance_count,
    )

    ml_client.online_deployments.begin_create_or_update(deployment).result()

    # Set deployment as default
    endpoint.traffic = {f"deployment-{endpoint_name}": 100}
    ml_client.online_endpoints.begin_create_or_update(endpoint).result()

    return ml_client.online_endpoints.get(name=endpoint_name)

def main():
    # 1. Initialize Azure ML workspace
    print("Initializing Azure ML workspace...")
    # NOTE: DefaultAzureCredential() will require `az login` to be run in the terminal via azure cli
    ml_client = MLClient.from_config(credential=DefaultAzureCredential())
    print("Azure ML workspace initialized!")

    # 2. Create an online endpoint
    endpoint_config = {
        "ml_client": ml_client,
        "endpoint_name": "hospital_readmissions_endpoint",
        "model_name": "hospital_readmissions_model",
        "model_version": "1",
        "instance_type": "Standard_DS3_v2",
        "instance_count": 1
    }
    endpoint = create_online_endpoint(**endpoint_config)
    print(f"Endpoint '{endpoint.name}' created successfully!")
    print(f"Scoring URI: {endpoint.scoring_uri}")

if __name__ == "__main__":
    main()
