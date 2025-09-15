from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
from joblib import load
import pathlib
from fastapi.middleware.cors import CORSMiddleware

origins = ["*"]

app = FastAPI(title="Fraud Detection API")

app.add_middleware(
   CORSMiddleware,
   allow_origins=origins,
   allow_credentials=True,
   allow_methods=["*"],
   allow_headers=["*"]
)

# Cargar modelo
model = load(pathlib.Path("model/fraud-detection-v1.joblib"))

# Definir entrada con todas las features relevantes
class InputData(BaseModel):
    Per1: float
    Per2: float
    Per3: float
    Per4: float
    Per5: float
    Per6: float
    Per7: float
    Per8: float
    Per9: float
    Dem1: float
    Dem2: float
    Dem3: float
    Dem4: float
    Dem5: float
    Dem6: float
    Dem7: float
    Dem8: float
    Dem9: float
    Cred1: float
    Cred2: float
    Cred3: float
    Cred4: float
    Cred5: float
    Cred6: float
    Normalised_FNT: float

class OutputData(BaseModel):
    score: float

@app.post("/score", response_model=OutputData)
def score(data: InputData):
    model_input = np.array([v for v in data.dict().values()]).reshape(1, -1)
    result = model.predict_proba(model_input)[:, -1][0]  # probabilidad de fraude
    return {"score": float(result)}
