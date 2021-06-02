import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from flask import pickle



# Load Sample data
data = pd.read_csv('dataset\iris.csv')
data.drop('Id', axis=1, inplace=True)

#Split loaded data into independent and target features
X = data.iloc[:,:-1] # independent variables
Y = data.iloc[:,-1]  # Target variable


#Train Logistic Regression model with all data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.30, random_state=101)
model = LogisticRegression()
model.fit(X_train, Y_train)

#Taking Predictions
predictions = model.predict(X_test)

#Save the model in the hard disk for future use
modelfile = open('Model\LRModel.pckl', 'wb')
pickle.dump(model, modelfile)
modelfile.close()

#Now your machine learning model is created and saved in hard-disk as SVMModel.pckl


































