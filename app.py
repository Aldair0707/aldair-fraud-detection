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


model = load(pathlib.Path("model/fraud-detection-v1.joblib"))

class InputData(BaseModel):
    Per1: float = 1.07
    Per2: float = 0.58
    Per3: float = 0.48
    Per4: float = 0.7667
    Per5: float = 1.2333
    Per6: float = 1.9933
    Per7: float = 0.34
    Per8: float = 1.01
    Per9: float = 0.8633
    Dem1: float = 0.46
    Dem2: float = 0.6433
    Dem3: float = 0.7367
    Dem4: float = 0.7567
    Dem5: float = 0.8133
    Dem6: float = 0.6933
    Dem7: float = 0.6667
    Dem8: float = 0.68
    Dem9: float = 0.7267
    Cred1: float = 0.6067
    Cred2: float = 1.01
    Cred3: float = 0.9333
    Cred4: float = 0.6033
    Cred5: float = 0.6867
    Cred6: float = 0.6733
    Normalised_FNT: float = -245.75


class OutputData(BaseModel):
    score: float

@app.post("/score", response_model=OutputData)
def score(data: InputData):
    model_input = np.array([v for v in data.dict().values()]).reshape(1, -1)
    result = model.predict_proba(model_input)[:, -1][0]  # probabilidad de fraude
    return {"score": float(result)}
