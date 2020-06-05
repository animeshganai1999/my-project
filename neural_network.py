#Artificial Neural Network
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:,3:13 ].values
y = dataset.iloc[:, 13].values

# Encoding categorical data
from sklearn.preprocessing import OneHotEncoder , LabelEncoder
from sklearn.compose import ColumnTransformer
labelencoder_X_1 = LabelEncoder() #Convert Strings into numbers
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
labelencoder_X_2 = LabelEncoder() #Convert Strings into numbers
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])
#Creating Dummy Variable for Country
ct = ColumnTransformer([('encoder',OneHotEncoder(),[1])],remainder = 'passthrough')
X = np.array(ct.fit_transform(X),dtype = np.float)
X = X[:,1:] #Removing one dummy variable to avoid dummy variable Trap

#spilliting the dataset into training set and test set
from sklearn.model_selection import train_test_split
#20% of dataset declear as test set
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2,random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#Now ANN Starts
#Importing keras libraries and required packages
import keras
from keras.models import Sequential #USe to initialise
from keras.layers import Dense

#Initialize The ANN
classifier = Sequential()

#Adding The Input layer and first Hidden Layer
classifier.add(Dense(output_dim = 6,init = 'uniform',activation = 'relu',input_dim = 11))

#Adding The Second Hidden Layer
classifier.add(Dense(output_dim = 6,init = 'uniform',activation = 'relu')) #We dont't need to specify input_dim ,coz it is already known

#Adding The Output Layer
classifier.add(Dense(output_dim = 1,init = 'uniform',activation = 'sigmoid'))

#Compiling th ANN
classifier.compile(optimizer = 'adam',loss = 'binary_crossentropy',metrics = ['accuracy'])

#Fitting The ANN to Our Training Set
classifier.fit(X_train,y_train,batch_size = 10,epochs = 100)

# Predicting the Test set results
y_pred = classifier.predict(X_test) #Returns probability whether the customer leave the bank or not

y_pred = (y_pred > 0.5) #return True or False
# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
