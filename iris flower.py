import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

data = pd.read_csv('E://New folder/iris.csv')
#print(data['Iris-setosa'].value_counts())
X = data.drop(['Id', 'Species'], axis=1)
y = data['Species']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=5)
knn = KNeighborsClassifier(n_neighbors=12)
knn.fit(X, y)
print("enter")
a=input().split()
print(knn.predict([a]))
#print(data.describe())