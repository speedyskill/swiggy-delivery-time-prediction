import dagshub
import mlflow
from mlflow.tracking import MlflowClient
import json

dagshub.init(repo_owner='speedyskill', repo_name='swiggy-delivery-time-prediction', mlflow=True)
mlflow.set_tracking_uri('https://dagshub.com/speedyskill/swiggy-delivery-time-prediction.mlflow')

def load_model_info(path):
    with open(path) as f:
        run_info = json.load(f)
    return run_info   

model_name = load_model_info('run_information.json')['model_name']
stage = 'Staging'

client = MlflowClient()

latest_version = client.get_latest_versions(name=model_name,stages=[stage])[0].version

promote_stage = 'Production'

client.transition_model_version_stage(
    name = model_name,
    version = latest_version,
    stage = promote_stage,
    archive_existing_versions = True
)
