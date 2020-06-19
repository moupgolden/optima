import pandas as pd 
import numpy as np
import pandas as pd
import pickle
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score 
from sklearn.metrics import classification_report 
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import linear_model
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier

#********************************************************************
def encode(data, col, max_val):
    data[col + '_sin'] = np.sin(2 * np.pi * data[col]/max_val)
    data[col + '_cos'] = np.cos(2 * np.pi * data[col]/max_val)
    data.drop([col],axis=1,inplace=True)
    return data
#**********************************************************************
tt = pd.read_csv("D:\plt\demand1.csv")
tt['status']=tt['status'].replace({2:3})


tt['date_time'] = tt['date_time'].apply(pd.to_datetime)

tt['year'] = [i.year for i in tt['date_time']]

tt['month'] = [i.month for i in tt['date_time']]

tt['day'] = [i.day for i in tt['date_time']]
tt['hour'] = [i.hour for i in tt['date_time']]
tt.drop('date_time',axis=1,inplace=True)
xt=tt[['hour','day','month','year']]
xt = encode(xt, 'month', 12)
xt = encode(xt, 'day', 30)
xt = encode(xt, 'hour', 24)
xt = xt[['day_sin','day_cos','month_sin','month_cos','year']]

yt=tt['status']
xt.head()
#**********************************
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(xt, yt, test_size = 0.2)
# summarize
print('Train', X_train.shape, Y_train.shape)
print('Test', X_test.shape, Y_test.shape)
#**********************************
lg = linear_model.LogisticRegression(max_iter=300)
lg.fit(X_train,Y_train).predict(X_test)

dtc = DecisionTreeClassifier()
dtc.fit(X_train,Y_train).predict(X_test)

rf = RandomForestClassifier(n_estimators = 100, random_state = 42)
rf.fit(X_train,Y_train).predict(X_test)

knn=KNeighborsClassifier()
knn.fit(X_train,Y_train).predict(X_test)

svm=SVC(probability=True)
svm.fit(X_train,Y_train).predict(X_test)

xgb_clf = XGBClassifier(gamma=2)#minimum loss reduction
xgb_clf.fit(X_train, Y_train)

models = (lg, dtc, rf, knn,svm,xgb_clf)
# Save tuple
pickle.dump(models, open("demand_models.pkl", 'wb'))

