import pandas as pd
import numpy as np
from sklearn.externals import joblib

# charger les données de test
x_test = pd.read_csv('xtest.csv')
y_test = pd.read_csv('ytest.csv')

# charger le modèle sauvgarder
nn = joblib.load("ann.sav")

# calculer et afficher le score
result = nn.score(x_test, y_test)
print("pourcentage prediction : %f" % (result*100))

# afficher les parametre du réseau de neurone
print("\nParamètres du réseau de neurones")
print(nn.get_params())
