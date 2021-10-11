# Prediction algorithm using ML

import pandas
from sklearn.tree import DecisionTreeClassifier

mobiledata=pandas.read_csv('mobilemodel.csv')
#print(mobiledata)

#training data
features=mobiledata.drop(columns=['mobile'])
labels=mobiledata['mobile']

#build a model
model=DecisionTreeClassifier()
model.fit(features,labels)

#test data and predict
result=model.predict([[25,2,35000],[40,1,75000]])
print(result)

