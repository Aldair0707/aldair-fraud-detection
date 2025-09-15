import pandas as pd
import requests

#cargar dataset
df = pd.read_csv("data/fraud-detection.csv")

df = df.drop(columns=["id", "Group", "Target"])

for i in range(5000):
    body = df.iloc[i].to_dict()
    response = requests.post("http://127.0.0.1:8000/score", json=body)
    print(f"Fila {i+1} → Score:", response.json())
