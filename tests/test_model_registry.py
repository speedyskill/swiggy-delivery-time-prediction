import pytest
import mlflow
from mlflow import MlflowClient
import dagshub
import json

dagshub.init(repo_owner='speedyskill', repo_name='swiggy-delivery-time-prediction', mlflow=True)
mlflow.set_tracking_uri('https://dagshub.com/speedyskill/swiggy-delivery-time-prediction.mlflow')

def load_model_info(path):
    with open(path) as f:
        run_info = json.load(f)

    return run_info        

model_name = load_model_info('run_information.json')['model_name']

@pytest.mark.parametrize(argnames='model_name,stage',argvalues=[(model_name,'Staging')])

def test_laod_model_from_registry(model_name,stage):
    client = MlflowClient()
    latest_versions = client.get_latest_versions(name=model_name,stages=[stage])
    latest_version=latest_versions[0].version if latest_versions else None

    assert latest_version is not None, f'No model at {stage} stage'

    model_uri = f'models:/{model_name}/{stage}'

    model = mlflow.sklearn.load_model(model_uri)

    assert model is not None, f'Failed to load model from Registry'
    print(f'The {model_name} model with version {latest_version} was loaded successfully')
