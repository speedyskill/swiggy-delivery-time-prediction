import pandas as pd
import requests
from pathlib import Path

# path for data
root_path = Path(__file__).parent.parent
data_path = root_path / "data" / "raw" / "swiggy.csv"

# prediction endpoint
predict_url = "http://127.0.0.1:8000/predict"

# sample row for testing
sample_row = pd.read_csv(data_path).dropna().sample(1)
print("The target value is", sample_row.iloc[:,-1].values.item().replace("(min) ",""))
    
# remove the target column
data = sample_row.drop(columns=[sample_row.columns.tolist()[-1]]).dropna().squeeze().to_dict()

# get the response from API
response = requests.post(url=predict_url,json=data)

print("The status code for response is", response.status_code)

if response.status_code == 200:
    result = response.json()
    print(f"Predicted time: {float(result['prediction']):.2f} min")
    print(f"Predicted distance: {float(result['distance']):.2f} km")
else:
    print("Error occurred:", response.text)