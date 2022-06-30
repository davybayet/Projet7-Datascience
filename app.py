import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from flask import Flask, jsonify, request
import pickle

app = Flask(__name__)
# charger le dataset et le model
FILE_TEST_SET = 'resources/data/test_set.pickle'
with open(FILE_TEST_SET, 'rb') as df_test_set:
            test_set = pickle.load(df_test_set)
FILE_BEST_MODELE = 'resources/modele/best_model.pickle'
with open(FILE_BEST_MODELE, 'rb') as model_lgbm:
            best_model = pickle.load(model_lgbm)

print("API ready")

@app.route('/')
def home():
    return 'Entrer un ID client dans la barre URL'

@app.route('/<int:client_id>/')
def predict(client_id):
    
    if client_id not in list(test_set['SK_ID_CURR']):
        result = 'Ce client n\'est pas dans la base de donnée'
    else:
        X_test=test_set[test_set['SK_ID_CURR']==int(client_id)]
        y_proba=best_model.predict_proba(X_test.drop(['SK_ID_CURR'],axis=1))[:, 1]
        result=('ce client est solvable avec un score de '+ str(np.around(y_proba*100,2))+'%')
    return result


if __name__ == '__main__':
    app.run(debug=True)
    pass