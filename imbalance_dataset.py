import pandas as pd
from sklearn.datasets import make_classification
import matplotlib.pyplot as plt

import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_curve,f1_score,precision_score,recall_score,accuracy_score,roc_auc_score


def roc_auc_curve(y_true,y_pred):
    fpr,tpr,threshold = roc_curve(y_true, y_pred)
    plt.plot(fpr,tpr)
    plt.xlabel('False Positive Ratio')
    plt.ylabel('True Positive Ratio')
    plt.show()
def precission_recall_curve_(y_true,y_pred):
    pre,rec,threshold = precision_recall_curve(y_true,y_pred)
    plt.plot(pre,rec)
    plt.xlabel('Precision')
    plt.ylabel('Recall')
    plt.show()
    
def model_report(y_true,y_pred):
    print("accuracy =",accuracy_score(y_true,y_pred))
    print("f1 score =",f1_score(y_true, y_pred))
    print('precission =',precision_score(y_true, y_pred))
    print('recall =',recall_score(y_true, y_pred))
    print('Roc_Auc score =',roc_auc_score(y_true, y_pred))
    
def grid_result(grid):
    print('best score =',grid.best_score_)
    print('best parameters =',grid.best_params_)
    
# define dataset
X, y = make_classification(n_samples=10000, weights=[0.99], flip_y=0)


#spliting the dataset
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.25)

from xgboost import XGBClassifier
clf = XGBClassifier()
clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)


roc_auc_curve(y_test,y_pred)
model_report(y_test, y_pred)
precission_recall_curve_(y_test, y_pred)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)















