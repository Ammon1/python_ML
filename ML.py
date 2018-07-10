# -*- coding: utf-8 -*-


import numpy as np
import pandas as pd

#import and save
aduld = pd.read_csv('adult.data')
X=aduld.iloc[:,:-1].values
Y=aduld.iloc[:,14].values

X[:,2]=X[:,2].astype('int')
X[:,0]=X[:,0].astype('int')
X[:,4]=X[:,4].astype('int')
X[:,10]=X[:,10].astype('int')
X[:,11]=X[:,11].astype('int')
X[:,12]=X[:,12].astype('int')

#encode Y
from sklearn.preprocessing import LabelEncoder
labelencoder_Y = LabelEncoder()
Y = labelencoder_Y.fit_transform(Y)

#change categories into separate columns
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
labelencoder_X = LabelEncoder()
X_gov=X[:,1:2]
X_gov[:,0] = labelencoder_X.fit_transform(X_gov[:,0])
onehotencoder = OneHotEncoder(categorical_features=[0])
X_gov=onehotencoder.fit_transform(X_gov).toarray()
X_gov=X_gov[:,:-1]

X_bach=X[:,3:4]
X_bach[:,0] = labelencoder_X.fit_transform(X_bach[:,0])
onehotencoder = OneHotEncoder(categorical_features=[0])
X_bach=onehotencoder.fit_transform(X_bach).toarray()
X_bach=X_bach[:,:-1]

X_married=X[:,5:6]
X_married[:,0] = labelencoder_X.fit_transform(X_married[:,0])
onehotencoder = OneHotEncoder(categorical_features=[0])
X_married=onehotencoder.fit_transform(X_married).toarray()
X_married=X_married[:,:-1]

X_adm=X[:,6:7]
X_adm[:,0] = labelencoder_X.fit_transform(X_adm[:,0])
onehotencoder = OneHotEncoder(categorical_features=[0])
X_adm=onehotencoder.fit_transform(X_adm).toarray()
X_adm=X_adm[:,:-1]

X_fam=X[:,7:8]
X_fam[:,0] = labelencoder_X.fit_transform(X_fam[:,0])
onehotencoder = OneHotEncoder(categorical_features=[0])
X_fam=onehotencoder.fit_transform(X_fam).toarray()
X_fam=X_fam[:,:-1]

X_race=X[:,8:9]
X_race[:,0] = labelencoder_X.fit_transform(X_race[:,0])
onehotencoder = OneHotEncoder(categorical_features=[0])
X_race=onehotencoder.fit_transform(X_race).toarray()
X_race=X_race[:,:-1]

X_sex=X[:,9:10]
X_sex[:,0] = labelencoder_X.fit_transform(X_sex[:,0])
onehotencoder = OneHotEncoder(categorical_features=[0])
X_sex=onehotencoder.fit_transform(X_sex).toarray()
X_sex=X_sex[:,:-1]

X_state=X[:,13:14]
X_state[:,0] = labelencoder_X.fit_transform(X_state[:,0])
onehotencoder = OneHotEncoder(categorical_features=[0])
X_state=onehotencoder.fit_transform(X_state).toarray()
X_state=X_state[:,:-1]

#join dataframes

X_result = np.concatenate([X_adm,X_bach,X_fam,X_gov,X_married,X_race,X_sex,X_state,],axis=1)
X1=X[:,0:1].astype('int')
X2=X[:,2:3].astype('int')
X3=X[:,4:5].astype('int')
X4=X[:,10:13].astype('int')
X_result1=np.concatenate([X_result,X1,X2,X3,X4,],axis=1)

#split data into train and test
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test=train_test_split(X_result1,Y,test_size=0.2,random_state=0)

#scale data
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.transform(X_test)

#KNN algorhytm
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors=5, metric='minkowski',p=2)
classifier.fit(X_train,y_train)

y_pred = classifier.predict(X_test)

#result error 21.6%
from sklearn.metrics import confusion_matrix
cm_KNN=confusion_matrix(y_test,y_pred)

#naive bayes
from sklearn.naive_bayes import GaussianNB, 
classifier = GaussianNB()
classifier.fit(X_train,y_train)
y_pred = classifier.predict(X_test)
cm_NB=confusion_matrix(y_test,y_pred)

#decision tree

from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion = 'entropy',random_state=0)
classifier.fit(X_train,y_train)
y_pred = classifier.predict(X_test)
#21.8% error
cm_TREE=confusion_matrix(y_test,y_pred)

#random forest

from sklearn.ensemble import RandomForestClassifier
classifier=RandomForestClassifier(n_estimators=10,criterion='gini',random_state=0)
classifier.fit(X_train,y_train)
y_pred = classifier.predict(X_test)
#17.3% error
cm_forest=confusion_matrix(y_test,y_pred)

#SVC
from sklearn.svm import SVC
classifier = SVC(kernel='rbf',random_state=0)
classifier.fit(X_train,y_train)
y_pred = classifier.predict(X_test)
#17.9% error
cm_SVC=confusion_matrix(y_test,y_pred)
