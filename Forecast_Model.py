#!/usr/bin/env python
# coding: utf-8

# In[315]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[316]:


import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import LinearSVR, SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
import seaborn as sns
from sklearn. linear_model import Lasso
import joblib

import warnings
warnings.filterwarnings(action='ignore')


# In[ ]:





# In[317]:


import pandas as pd
data=pd.read_csv('https://raw.githubusercontent.com/Chitkara-University-Pb/Binary-Brains/main/ModifiedDatasetTomato.csv')
data.head()


# In[318]:


#encoding season column
data.replace({'Season':{'winter':1,'summer':2,'monsoon':3}},inplace=True)
data.replace({'Month':{ 
        'march': 3,
        'april': 4,
        'may': 5,
        'june': 6,
        'july': 7,
        'august': 8}},inplace=True)
              
data.head()


# In[319]:


#splitting dataset into X and y
y = data['avg price']
X = data.drop(['avg price','prod_name', 'order', 'prod_id', 'pack sold'], axis=1)
#for demand
z = data['order']
w = data.drop(['avg price', 'order', 'pack sold' , 'prod_id', 'prod_name'], axis =1)


# In[320]:


#splitting into testing and trainning
X_train,X_test,y_train,y_test=train_test_split(X,y,train_size=0.6,shuffle=True,random_state=1)
X_train,X_test,y_train,y_test=train_test_split(X,y,train_size=0.6,shuffle=True,random_state=1)
w_train,w_test,z_train,z_test=train_test_split(w,z,train_size=0.6,shuffle=True,random_state=1)
w_train,w_test,z_train,z_test=train_test_split(w,z,train_size=0.6,shuffle=True,random_state=1)


# In[321]:


models = {
    "                     Linear Regression": LinearRegression(),
    " Linear Regression (L2 Regularization)": Ridge(),
    " Linear Regression (L1 Regularization)": Lasso(),
    "                   K-Nearest Neighbors": KNeighborsRegressor(),
    "                        Neural Network": MLPRegressor(),
    "Support Vector Machine (Linear Kernel)": LinearSVR(),
    "   Support Vector Machine (RBF Kernel)": SVR(),
    "                         Decision Tree": DecisionTreeRegressor(),
    "                         Random Forest": RandomForestRegressor(),
    "                     Gradient Boosting": GradientBoostingRegressor(),
    "                               XGBoost": XGBRegressor(),
    "                              LightGBM": LGBMRegressor(),
    "                              CatBoost": CatBoostRegressor(verbose=0)
}

for name, model in models.items():
    model.fit(X_train, y_train)
    print(name + " trained.")

for name, model in models.items():
    print(name + " R^2 Score: {:.5f}".format(model.score(X_test, y_test)))


# In[322]:


#model selection linear regression
price_reg_model= CatBoostRegressor(verbose=0)
price_reg_model.fit(X_train,y_train)

order_reg_model = CatBoostRegressor(verbose=0)
order_reg_model.fit(w_train,z_train)


# In[323]:


#Training data prediction
price_data_pred= price_reg_model.predict(X_train)


# In[324]:


order_data_pred= order_reg_model.predict(w_train)


# In[325]:


from sklearn import metrics
error_score_price = metrics.r2_score(y_train,price_data_pred )
print (error_score_price)


# In[326]:


error_score_order = metrics.r2_score(z_train,order_data_pred )
print (error_score_order)


# In[327]:


from sklearn.metrics import mean_squared_error
print(mean_squared_error(y_train, price_data_pred))
print(mean_squared_error(z_train, order_data_pred))


# In[328]:


#visualize the actual prices and Predicted prices
from matplotlib import pyplot as plt
#plt.scatter(X_train,y_train, color = 'blue', label = 'Actual')
plt.scatter(y_train,price_data_pred)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual Price Vs Predicted Price")
plt.grid(True)
plt.show()


# In[329]:


plt.scatter(z_train,order_data_pred)
plt.xlabel("Actual Demand")
plt.ylabel("Predicted Demand")
plt.title("Actual Demand Vs Predicted Demand")
plt.grid(True)
plt.show()


# In[330]:


#Training data prediction
price_testing_data_pred= price_reg_model.predict(X_test)


# In[331]:


order_testing_data_pred= order_reg_model.predict(w_test)


# In[332]:


from sklearn import metrics
error_score_price_test = metrics.r2_score(y_test,price_testing_data_pred )
print (error_score_price_test)
error_score_order_test = metrics.r2_score(z_test,order_testing_data_pred )
print (error_score_order_test)


# In[333]:


from sklearn.metrics import mean_squared_error
print(mean_squared_error(y_test, price_testing_data_pred))
print(mean_squared_error(z_test, order_testing_data_pred))


# In[334]:


#visualize the actual prices and Predicted prices
from matplotlib import pyplot as plt
#plt.scatter(X_train,y_train, color = 'blue', label = 'Actual')
plt.scatter(y_test,price_testing_data_pred)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual Price Vs Predicted Price")
plt.grid(True)
plt.show()


# In[335]:


plt.scatter(z_test,order_testing_data_pred)
plt.xlabel("Actual Demand")
plt.ylabel("Predicted Demand")
plt.title("Actual Demand Vs Predicted Demand")
plt.grid(True)
plt.show()


# In[336]:


input_data = pd.read_csv('https://raw.githubusercontent.com/Chitkara-University-Pb/Binary-Brains/main/input.csv')


# In[337]:


inputPred = input_data.drop([], axis =1)


# In[338]:


price_predict_input = price_reg_model.predict(inputPred) #date, season , month
order_predict_input = order_reg_model.predict(inputPred) #Avg Price, date, season , month
price_predict_input = pd.DataFrame(price_predict_input, columns=['predicted avg price']).to_csv('priceForecast.csv')
order_predict_input = pd.DataFrame(order_predict_input, columns=['predicted demand']).to_csv('orderForecast.csv')


# In[339]:


# Saving Machine Learning Model
import joblib
cb = CatBoostRegressor(verbose=0)
joblib.dump(cb,"PriceOrderForecast.pkl")


# In[340]:


model = joblib.load("PriceOrderForecast.pkl")


# In[ ]:





# In[ ]:




