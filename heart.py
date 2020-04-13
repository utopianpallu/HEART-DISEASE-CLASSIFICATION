

import numpy as np
import pandas as pd
df = pd.read_csv("/Users/pallavi385/Downloads/heart.csv")
x=df.iloc[:,:-1].values
y=df.iloc[:,13].values
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.2,random_state=0)
from sklearn.naive_bayes import GaussianNB
nb = GaussianNB()
nb.fit(x_train, y_train)
nb_pred=nb.predict(x_test)
pred=pd.DataFrame(nb_pred, columns=['predictions'])
