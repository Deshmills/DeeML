from typing import List
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
from model import load_model

app = FastAPI(
    title='Bank Note Authenticator API',
    description='A simple API for verification of bank note',
    version='1.0.0'
)

model = load_model()


class PredictAuthenticationRequest(BaseModel):
    data: List[List[float]]


class PredictionAutenticationResponse(BaseModel):
    data: List[float]


@app.post('/predict', response_model=PredictionAutenticationResponse, tags=['prediction'])
async def predict(input: PredictAuthenticationRequest):
    X = np.array(input.data)
    y_pred = model.predict(X)
    result = PredictionAutenticationResponse(data=y_pred.tolist())
    return result


@app.get('/healthcheck', status_code=200, tags=['health_check'])
async def healthcheck():
    return 'Bank Node Authentication Classifier is ready my guy'
