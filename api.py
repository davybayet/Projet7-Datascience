import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from flask import Flask, jsonify
import pickle
import requests

app = Flask(__name__)
# charger le dataset et le model
FILE_TEST_SET = 'resources/data/test_set.pickle'
with open(FILE_TEST_SET, 'rb') as df_test_set:
            test_set = pickle.load(df_test_set)
FILE_BEST_MODELE = 'resources/modele/best_model.pickle'
best_model = pickle.load(model_lgbm)

print("API ready")

def predict(client_id):
    X_test = test_set[test_set['SK_ID_CURR'] == client_id]
    # Score des prédictions de probabiltés
    y_proba = best_model.predict_proba(X_test.drop('SK_ID_CURR', axis=1))[:, 1]
    return y_proba

@app.route("/")
def super_endpoint():
    return "Projet 7 API"

@app.route("/predict")
def predict():
    client_id = request.args.get('client_id')
    # verifier le type de client_id si besoin en faire un int
    proba = predict(client_id)
    
    return jsonify(proba)


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5002, debug=True)
    pass
