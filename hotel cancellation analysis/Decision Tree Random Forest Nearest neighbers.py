#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
hotel=pd.read_csv('hotel_booking_data_cleaned.csv') #load data as pandas dataframe

# Based on the conclusion of the descriptive analysis just now, we preprocess the dataset as follows

# 删除 adr<0 和 adr>5000 的行  
hotel = hotel[(hotel['adr'] >= 0) & (hotel['adr'] <= 5000)]  
 
nan_replacements = {"children": 0 }
#替换缺失项得到新数据
hotel_cln = hotel.fillna(nan_replacements)

#替换full_data_cln中不规范值
#meal字段包含'Undefined'意味着自带食物SC
#关于meal字段缩写代表的意义，########333
hotel_cln["meal"].replace("Undefined", "SC", inplace=True)
hotel_cln["market_segment"].replace("Undefined", "Complementary", inplace=True)
hotel_cln["distribution_channel"].replace("Undefined", "Direct", inplace=True)
print(hotel_cln) 


# In[3]:


from sklearn.model_selection import train_test_split # this function provides a single "Hold-Out" Validation.
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score #similar to MAE, we use accuracy_score evaluation metric.
import pandas as pd
import numpy as np
from sklearn.svm import SVC 
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis


# In[4]:


hotel_cln.isnull().sum()


# In[5]:


from sklearn.preprocessing import LabelEncoder
categorical_columns = hotel_cln.select_dtypes(include=['object']).columns
label_encoder = LabelEncoder()
for column in categorical_columns:
    hotel_cln[column]= label_encoder.fit_transform(hotel_cln[column])

print (hotel_cln)


# In[6]:


hotel_cln.isnull().sum()


# In[8]:


y = hotel_cln['is_canceled'] #this is our prediction target
X = hotel_cln.drop(['is_canceled'],axis=1)
#convert categorical into numerical 
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.3, random_state=0)


# ### Decision Tree

# ### 超参数调优

# In[51]:


from sklearn.model_selection import GridSearchCV 
  
# defining hyperparameter options 
param_grid = {'max_depth': [3, 5, 10], 'min_samples_split': [2, 5, 10], 'min_samples_leaf': [1, 2, 4]  
              }  
  
grid = GridSearchCV(DecisionTreeClassifier(random_state=0), param_grid, refit = True, verbose = 3) 
  
# fitting the model for grid search 
grid.fit(X_train, Y_train) 


# In[52]:


# print best parameter after tuning 
print(grid.best_params_) 
  
# print how our model looks after hyper-parameter tuning 
print(grid.best_estimator_)


# In[53]:


from sklearn.model_selection import cross_val_score
clf= DecisionTreeClassifier(max_depth=15,min_samples_split=2,min_samples_leaf=1)
scores = cross_val_score(clf, X, y, cv=5)
print('DT CV Score:', scores)# cross validation


# In[54]:


from sklearn.metrics import classification_report # this library directly generates precision, recall, f-measure
from sklearn.model_selection import train_test_split

clf= DecisionTreeClassifier(max_depth=10,min_samples_split=2,min_samples_leaf=1)
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=0)
clf.fit(X_train,y_train)
y_pred=clf.predict(X_test)

print(classification_report(y_test, y_pred))


# ### RandomForest

# In[95]:


from sklearn.model_selection import GridSearchCV 
  
# defining hyperparameter options 
param_grid = {'max_depth': [2,5,8,10],  
              'n_estimators': [10,50,100], 
              'max_features': [2,4,6]}  
  
grid = GridSearchCV(RandomForestClassifier(random_state=0), param_grid, refit = True, verbose = 3) 
  
# fitting the model for grid search 
grid.fit(X_train, Y_train) 


# In[39]:


# print best parameter after tuning 
print(grid.best_params_) 
  
# print how our model looks after hyper-parameter tuning 
print(grid.best_estimator_)


# In[40]:


from sklearn.model_selection import cross_val_score
clf=RandomForestClassifier(max_depth=10, n_estimators=50, max_features=6)
scores = cross_val_score(clf, X, y, cv=5)
print('RF CV Score:', scores)# cross validation


# In[42]:


from sklearn.metrics import classification_report # this library directly generates precision, recall, f-measure
from sklearn.model_selection import train_test_split

clf=RandomForestClassifier(max_depth=10,n_estimators=50, max_features=6)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
clf.fit(X_train,y_train)
y_pred=clf.predict(X_test)

print(classification_report(y_test, y_pred))


# ### KNN

# In[9]:


from sklearn.model_selection import GridSearchCV 
  
# defining hyperparameter options 
param_grid = {'n_neighbors': [2,5,8,10],  
             }  
  
grid = GridSearchCV(KNeighborsClassifier(), param_grid, refit = True, verbose = 3) 
  
# fitting the model for grid search 
grid.fit(X_train, Y_train) 


# In[10]:


# print best parameter after tuning 
print(grid.best_params_) 
  
# print how our model looks after hyper-parameter tuning 
print(grid.best_estimator_)


# In[11]:


from sklearn.model_selection import cross_val_score
clf=KNeighborsClassifier(2)
scores = cross_val_score(clf, X, y, cv=5)
print('RF CV Score:', scores)# cross validation


# In[22]:


from sklearn.metrics import classification_report # this library directly generates precision, recall, f-measure
from sklearn.model_selection import train_test_split

clf=KNeighborsClassifier(2)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
clf.fit(X_train,y_train)
y_pred=clf.predict(X_test)

print(classification_report(y_test, y_pred))


# In[26]:


clf4 = RandomForestClassifier(max_depth=10,n_estimators=50, max_features=6)
clf4.fit(X_train, y_train)
Y_predTrain4 = clf4.predict(X_train)
Y_predTest4= clf4.predict(X_test)

clf5 = KNeighborsClassifier(2)
clf5.fit(X_train, y_train)
Y_predTrain5 = clf5.predict(X_train)
Y_predTest5 = clf5.predict(X_test)

fig, axs = plt.subplots(ncols=2, figsize=(10, 3.9))

ConfusionMatrixDisplay.from_predictions(y_test, Y_predTest4, labels=clf4.classes_, cmap= plt.cm.BuGn, ax=axs[0])
rfm = ConfusionMatrixDisplay.from_predictions(y_test, Y_predTest5, labels=clf5.classes_, cmap= plt.cm.Reds, ax=axs[1])
axs[0].set_title("Random Forest")
axs[1].set_title("KNN")

print("------------------------Random Forest--------------- ---------------")
print(classification_report(y_test, Y_predTest4))
print("------------------------KNN-------------------------------")
print(classification_report(y_test, Y_predTest5))


# In[27]:


scores = cross_val_score(clf4, X, y, cv=5)
print('Random Forest CV Score:', scores)
scores = cross_val_score(clf5, X, y, cv=5)
print('KNN CV Score:', scores)


# ### PCA feature dimension

# In[35]:


import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
pca = PCA(n_components=0.95) 
z = pca.fit(X).transform(X)
display(z[:5,:])


# In[36]:


clf=KNeighborsClassifier(2)
scores = cross_val_score(clf, z,y,cv=5)
print('RF CV Score:', scores)# cross validation


# ### Cross Validation

# In[43]:


h = 0.02  # step size in the mesh

names = [
    "Nearest Neighbors",
    "Decision Tree",
    "Random Forest",
]

classifiers = [
    KNeighborsClassifier(2),
    DecisionTreeClassifier(max_depth=10,min_samples_split=2,min_samples_leaf=1),
    RandomForestClassifier(max_depth=10, n_estimators=50, max_features=6),
]


# In[44]:


for i in range(len(classifiers)):
    clf=classifiers[i] #use the i-th model in the "classifiers" list
    scores = cross_val_score(clf, X, y, cv=5)
    print('CV Score of '+ names[i], scores)


# In[ ]:





# In[ ]:




