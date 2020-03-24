import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.model_selection import cross_val_score

# lien documentation : https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html

# charger les données
data = pd.read_csv('caract.csv')

df_x = data.iloc[:, 0:19]
df_y = data.iloc[:, -1]

# diviser les données en jeux d'entrainement et de teste
# les données de teste representent 20% de tous les données dans le fichier caract.csv

for i in range(10, 20):
    print("\n====== %d ========\n" % i)
    x_train, x_test, y_train, y_test = train_test_split(
        df_x, df_y, test_size=0.20, random_state=i)

    # creé le svm
    clf = svm.SVC(kernel='linear', C=20).fit(x_train, y_train)
    # print(y_test.values)
    print(clf.score(x_test, y_test))
