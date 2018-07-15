import pandas
import numpy as xppp
from pandas.plotting import scatter_matrix
import dataset
import matplotlib.pyplot as plt
import math
import random

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.linear_model import LinearRegression
from sklearn import ensemble
from sklearn.metrics import mean_squared_error
from math import sqrt

#Here Data's to be trained are inputted 

Trainer_datas = pandas.read_csv("apartment.csv")
The_set=Trainer_datas.describe()
print(The_set)
#in this condition the unnecessary data values are ignored by dropiing
A  = Trainer_datas.drop(['id','price','date','waterfront','long'],axis=1)
B = Trainer_datas['price']

A = A.astype('int')
B = B.astype('int')

A_tr, A_varify, B_tr, B_varify = model_selection.train_test_split(A, B, test_size=.3, random_state=2)

#Those are the algorithms we will implement here
algorithms = []
algorithms.append(('Linear_Regression', LinearRegression()))
algorithms.append(('Gradient',ensemble.GradientBoostingRegressor()))



# Each algorithm will be run 



for n, algo in algorithms:
    algo.fit(A_tr, B_tr)
    print(n,':')
    print(algo.score(A_varify, B_varify))
#accuracy 64 r 79

type1 = LinearRegression()
type1.fit(A_tr, B_tr)

type2=ensemble.GradientBoostingRegressor();
type2.fit(A_tr, B_tr)



result1 = type1.predict([[4,2,1200,5500,1,0,5,5,1200,0,1970,0,47,98178,1300,5000]])
result2 = type2.predict([[4,2,1200,5500,1,0,5,5,1200,0,1970,0,47,98178,1300,5000]])


print('Result of this new data : ',result1, "Bdt")

print('Result of this new data : ',result2, "Bdt")












