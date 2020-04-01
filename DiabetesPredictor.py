# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import matplotlib.pyplot as plt
col_names = ['pregnant', 'glucose', 'bp', 'skin', 'insulin', 'bmi', 'pedigree', 'age', 'label']
# load dataset
diab = pd.read_csv("diabetes.csv", header=0, names=col_names)

#split dataset in features and target variable
feature_cols = ['pregnant', 'insulin', 'bmi', 'age','glucose','bp','pedigree']
for col in feature_cols:
	diab[col] = diab[col].astype(dtype=np.float64)

X = diab[feature_cols]
y = diab.label

#creating train and test sets
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=0)

#training algo
logreg = LogisticRegression()

# fit the model with data
logreg.fit(X_train, y_train)

#predict on test set
y_pred=logreg.predict(X_test)
print(logreg.predict_proba(X_test)) #probabilities
print(y_pred) #binary labelling

#plotting a piechart
count0=0
count1=0
for i in y_pred:
    if i==1:
        count1=count1+1
    else:
        count0=count0+1
sizes=[count0,count1]        
labels='healthy' , 'diabetic'
colors=['gold' , 'lightcoral']
explode = (0.1, 0)
plt.pie(sizes, explode=explode, labels=labels, colors=colors,autopct='%1.1f%%', shadow=True)        

#creating confusion matrix to evaluate performance of our model
c_matrix = metrics.confusion_matrix(y_test, y_pred)
c_matrix

#evaluation metrics
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print("Precision:",metrics.precision_score(y_test, y_pred))
print("Recall:",metrics.recall_score(y_test, y_pred))