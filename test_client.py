import pandas as pd
import requests

# Cargar dataset
df = pd.read_csv("data/fraud-detection.csv")

# Eliminar columnas que no se usan
df = df.drop(columns=["id", "Group", "Target"])

# Probar con las primeras 5 filas
for i in range(5000):
    body = df.iloc[i].to_dict()
    response = requests.post("http://127.0.0.1:8000/score", json=body)
    print(f"Fila {i+1} → Score:", response.json())
