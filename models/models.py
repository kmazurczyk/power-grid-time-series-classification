from dotenv import load_dotenv
import os
import numpy as np
import pandas as pd
from __init__ import get_base_path
from sklearn.metrics import accuracy_score

class Model:
    def __init__(self, model, config: dict):
        self.model = model
        self.metrics = []
        self.config = config
    
    def __str__(self):
        return f"Model: {self.model} configured with {self.config} metrics {self.metrics}"

    def train(self, X, y) -> None:
        return self.model.fit(X)

    def predict(self, X) -> np.array:
        return self.model.predict(X)

    def score(self, X, y):
        return accuracy_score(y, self.predict(X))

def model_runner(data, model, config):
    pass
    # feature selection
    # data split
    # pre-process
    # train
    # cross_val / tune
    # evaluate

if __name__ == "__main__":
    
    pass 
