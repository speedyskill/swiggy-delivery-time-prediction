import pytest
import mlflow
from mlflow.tracking import MlflowClient
import json
from sklearn.pipeline import Pipeline
import joblib
import pandas as pd
from sklearn.metrics import mean_absolute_error,r2_score
import dagshub

dagshub.init(repo_owner='speedyskill', repo_name='swiggy-delivery-time-prediction', mlflow=True)
mlflow.set_tracking_uri('https://dagshub.com/speedyskill/swiggy-delivery-time-prediction.mlflow')

def load_model_info(path):
    with open(path) as f:
        run_info = json.load(f)

    return run_info    

def load_transformer(path):
    transformer = joblib.load(path)
    return transformer

model_name = load_model_info('run_information.json')['model_name']
stage = 'Staging'

model_uri = f'models:/{model_name}/{stage}'

model = mlflow.sklearn.load_model(model_uri)

transformer = load_transformer('models/preprocessor.joblib')

model_pipe = Pipeline(steps=[
    ('preprocessor', transformer),
    ('regressor', model)
])

test_data_path = 'data/interim/test.csv'

@pytest.mark.parametrize(argnames='model_pipe, test_data_path, threshold_error',
                         argvalues=[(model_pipe,test_data_path,5)])

def test_model_performance(model_pipe,test_data_path,threshold_error):
    
    df = pd.read_csv(test_data_path)

    df.dropna(inplace=True)

    X = df.drop(columns='time_taken')
    y = df['time_taken']

    y_pred = model_pipe.predict(X)

    mean_error = mean_absolute_error(y,y_pred)

    assert mean_error <= threshold_error, f'The model failed to pass the performance test'
    
    print('Average error is ',mean_error)

    print(f'The {model_name} model passed the performance test')