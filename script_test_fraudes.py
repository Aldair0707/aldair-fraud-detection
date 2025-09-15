import pandas as pd
import requests

# Cargar dataset
df = pd.read_csv("data/fraud-detection.csv")

# Filtrar solo filas con fraude (Target = 1)
fraudes = df[df["Target"] == 1].drop(columns=["id", "Group", "Target"])

if fraudes.empty:
    print("⚠️ No se encontraron casos de fraude en el dataset.")
else:
    # Probar con los primeros 5 fraudes
    for i in range(min(50, len(fraudes))):
        body = fraudes.iloc[i].to_dict()
        response = requests.post("http://127.0.0.1:8000/score", json=body)
        
        if response.status_code == 200:
            result = response.json()
            score = result.get("score", 0.0)
            detected = " Detectado" if score >= 0.5 else " No detectado"
            print(f"Fraude {i+1} → Score: {score:.4f} → {detected}")
        else:
            print(f"Error en la petición: {response.status_code}, {response.text}")
