import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_boston
boston = load_boston()
df = pd.DataFrame(boston.data)
df.columns = boston.feature_names
df['price'] = boston.target
X,y = df.iloc[:,:-1],df.iloc[:,-1]

import xgboost as xgb
data_matrics = xgb.DMatrix(data = X,label = y)

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.25)

regressor = xgb.XGBRegressor(objective = 'reg:linear',
                               colsample_bytree = 0.3,learning_rate = 0.1,
                               max_depth = 5,alpha = 10,n_estimators = 100)
regressor.fit(X_train,y_train)
y_pred = regressor.predict(X_test)

from sklearn.metrics import mean_squared_error,r2_score

print("R square error =",r2_score(y_test,y_pred))
print("mean square error =",mean_squared_error(y_test,y_pred))
