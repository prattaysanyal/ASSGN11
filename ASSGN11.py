from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sn
import pandas as pd
import numpy as np
%matplotlib inline
#Question 1
from sklearn.datasets import load_digits
# Question 2
digit=load_digits()
x=digit.data
y=digit.target
x.shape,y.shape #64 columns
plt.imshow(x[4].reshape(8,8),cmap=plt.cm.gray)
#Question 3
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.3)
# Question 4 
from sklearn.neighbors import KNeighborsClassifier
model=KNeighborsClassifier()
model.fit(x_train,y_train)
pred = model.predict(x_test)
# Question 5
from sklearn import model_selection
kfold=model_selection.KFold(n_splits=10,random_state=7)
results=model_selection.cross_val_score(model,x,y,cv=kfold,scoring="accuracy")
results
results.sum()/10
from sklearn.metrics import classification_report
print(classification_report(y_test,pred))
data1 = pd.DataFrame({"Predicted":pred,"Actual":y_test})
data1

