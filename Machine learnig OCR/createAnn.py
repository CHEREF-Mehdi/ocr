import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib

# charger les données
data = pd.read_csv('caract.csv')

df_x = data.iloc[:, 0:19]
df_y = data.iloc[:, -1]

# diviser les données en jeux d'entrainement et de teste
# les données de teste represente 20% de tous les données dans le fichier caract.csv
x_train, x_test, y_train, y_test = train_test_split(
    df_x, df_y, test_size=0.20, random_state=2)

# créer l'architecture de notre réseau de neurone
nn = MLPClassifier(activation='tanh', solver='adam', max_iter=450,
                   alpha=1e-30, hidden_layer_sizes=(14, 14), random_state=13)

# entrainer les réseau de neurone
nn.fit(x_train, y_train)

# prediction sur les donnée de test
pred = nn.predict(x_test)
a = y_test.values

# calculer et afficher le score
result = nn.score(x_test, y_test)
print("pourcentage prediction : ", result*100)

# sauvegarder les données de test
np.savetxt("xtest.csv", x_test, delimiter=",")
np.savetxt("ytest.csv", y_test.astype(int), fmt='%i', delimiter=",")
# sauvegarder le reseau de neurone
joblib.dump(nn, "ann_2.sav")
