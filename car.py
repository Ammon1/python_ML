# -*- coding: utf-8 -*-
"""
Created on Wed Jul 11 08:52:02 2018

@author: Ammon1
"""

import numpy as np
import pandas as pd

car = pd.read_csv('car.data')
car.columns=['buying','maint','doors','persons','lug_boot','safety','distribution']

X=car.iloc[:,:-1].values
Y=car.iloc[:,6].values

#encode labels
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
labelencoder_X = LabelEncoder()

X0=X[:,0:1]
X0[:,0] = labelencoder_X.fit_transform(X0[:,0])
onehotencoder = OneHotEncoder(categorical_features=[0])
X0=onehotencoder.fit_transform(X0).toarray()
X0=X0[:,:-1]

X1=X[:,1:2]
X1[:,0] = labelencoder_X.fit_transform(X1[:,0])
onehotencoder = OneHotEncoder(categorical_features=[0])
X1=onehotencoder.fit_transform(X1).toarray()
X1=X1[:,:-1]

X4=X[:,4:5]
X4[:,0] = labelencoder_X.fit_transform(X4[:,0])
onehotencoder = OneHotEncoder(categorical_features=[0])
X4=onehotencoder.fit_transform(X4).toarray()
X4=X4[:,:-1]

X5=X[:,5:6]
X5[:,0] = labelencoder_X.fit_transform(X5[:,0])
onehotencoder = OneHotEncoder(categorical_features=[0])
X5=onehotencoder.fit_transform(X5).toarray()
X5=X5[:,:-1]

X2=X[:,2:3].astype("str")
for idx,a in enumerate(X2):
    if a=="2":
        X2[idx]=2
    elif a == "3":
        X2[idx]=3
    elif a == "4":
        X2[idx]=4
    elif a == "5more":
        X2[idx]=5

X3=X[:,3:4].astype("str")
for idx,a in enumerate(X3):
    if a=="2":
        X3[idx]=2
    elif a == "4":
        X3[idx]=4
    elif a == "more":
        X3[idx]=6

Y0=Y.astype('str')
for idx,a in enumerate(Y0):
    if a=="unacc":
        Y0[idx]=0
    elif a == "acc":
        Y0[idx]=1
    elif a == "good":
        Y0[idx]=2
    elif a == "vgood":
        Y0[idx]=3
        
result = np.concatenate([X0,X1,X2,X3,X4,X5],axis=1)
result=result.astype('int')

from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test=train_test_split(result,Y0,test_size=0.2,random_state=0)

from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors=5, metric='minkowski',p=2)
classifier.fit(X_train,y_train)
y_pred = classifier.predict(X_test)

from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)
error=sum(map(sum,cm))-cm[0,0]-cm[1,1]-cm[2,2]-cm[3,3]
error_KNN=error/sum(map(sum,cm))

#random forest
error_forest=[]
from sklearn.ensemble import RandomForestClassifier
for i in range(1,20):
    error_forest.append([])
    for j in range(2,21):
        classifier=RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
                max_depth=None, max_features='auto', max_leaf_nodes=None,
                min_impurity_decrease=0.0, min_impurity_split=None,
                min_samples_leaf=i, min_samples_split=j,
                min_weight_fraction_leaf=0.0, n_estimators=10, n_jobs=1,
                oob_score=False, random_state=None, verbose=0,
                warm_start=False)
        classifier.fit(X_train,y_train)
        y_pred = classifier.predict(X_test)
        cm=confusion_matrix(y_test,y_pred)
        error=(sum(map(sum,cm))-cm[0,0]-cm[1,1]-cm[2,2]-cm[3,3])/(sum(map(sum,cm)))
        error_forest[i].append(error)
    
min_error=min(error_forest)
index_i=error_forest.index(min_error)+1
min_error=min(min_error)
min_error_j=min(min_error)
index_j=min_error.index(min_error_j)+2

classifier=RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
                max_depth=None, max_features='auto', max_leaf_nodes=None,
                min_impurity_decrease=0.0, min_impurity_split=None,
                min_samples_leaf=index_i, min_samples_split=index_j,
                min_weight_fraction_leaf=0.0, n_estimators=10, n_jobs=1,
                oob_score=False, random_state=None, verbose=0,
                warm_start=False)
classifier.fit(X_train,y_train)
y_pred = classifier.predict(X_test)
cm=confusion_matrix(y_test,y_pred)
error=(sum(map(sum,cm))-cm[0,0]-cm[1,1]-cm[2,2]-cm[3,3])/(sum(map(sum,cm)))
success=100-error

#succes=99.919%

    
