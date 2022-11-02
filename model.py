import pickle
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor

MODEL_ID = '6de767d1-596a-4c1f-ad71-976d05e698f8'
MODEL_PATH = Path().resolve()/f'{MODEL_ID}/{MODEL_ID}.pkl'


def load_model():
    with open(MODEL_PATH, 'rb') as rf:
        model = pickle.load(rf)
    return model
