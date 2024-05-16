#!/usr/bin/env python
# coding: utf-8

# In[176]:


import pandas as pd
import numpy as np
from sklearn import metrics
from scipy.stats import boxcox
from statsmodels.tsa.stattools import kpss
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from matplotlib import pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# # 0.Process data to calculate daily cancellation rate

# In[186]:


import random
from matplotlib.pylab import rcParams

df = pd.read_csv('hotel_booking_data_cleaned.csv', 
                 usecols=['is_canceled', 'arrival_date_year',  'arrival_date_month', 'arrival_date_day_of_month'])

month_map = {
    'January': 1, 'February': 2, 'March': 3, 'April': 4, 'May': 5, 'June': 6,
    'July': 7, 'August': 8, 'September': 9, 'October': 10, 'November': 11, 'December': 12
}
df['arrival_date_month'] = df['arrival_date_month'].map(month_map)

df['date'] = df['arrival_date_year'].astype(str) + '-' + df['arrival_date_month'].astype(str).str.zfill(2)+'-'+df['arrival_date_day_of_month'].astype(str).str.zfill(2)

# Convert 'date' column to datetime type and sort the entire DataFrame
df['date'] = pd.to_datetime(df['date'])
df.sort_values('date', inplace=True)

date_array = df['date'].unique().tolist()
grouped = df.groupby(['date', 'is_canceled']).size()

count_0 = []
count_1 = []
cancellation = []

# Traverse each date and is_canceled status and extract the corresponding count
for (date, is_canceled), count in grouped.items():
    total = len(df[df['date'] == date])
    if total == 0:
        cancellation.append(0)
    else:
        filtered_0 = df[(df['date'] == date) & (df['is_canceled'] == 0)]
        filtered_1 = df[(df['date'] == date) & (df['is_canceled'] == 1)]
        count_0.append(filtered_0.shape[0])  
        count_1.append(filtered_1.shape[0])

for i in range(0,len(date_array)):
       cancellation.append(count_1[i]/(count_1[i]+count_0[i]))
    

print("cancel rate:", cancellation)

    
len(cancellation)


# In[187]:


plt.figure(figsize = (20,8))
plt.plot(date_array, cancellation)
plt.title('Daily cancel rate')


# # Divide training set and test set. Total 793 days, forecast last 14 days

# In[195]:


from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

Y = cancellation_BC_deseason   #this is our prediction target
X = date_array
X_train = date_array[:-14]
Y_train = cancellation_BC_deseason[:-14]
X_test = date_array[-14:]
Y_test = cancellation_BC_deseason[-14:]


# In[196]:


plt.figure(figsize = (20,8))
plt.plot(X_train, Y_train, label = 'Train')
plt.plot(X_test, Y_test, label = 'Test')
plt.legend(loc = 'upper left')
plt.title('Daily cancel rate')


# # 1.Baseline model : ETS

# In[206]:


from statsmodels.tsa.exponential_smoothing.ets import ETSModel
cr = ETSModel(Y_train, trend=None)
cr_fit = cr.fit()
y_predict_3 = cr_fit.forecast(14)


# In[207]:


plt.figure(figsize = (20,8))
plt.plot(X_train, Y_train, label = 'Train')
plt.plot(X_test, Y_test, label = 'Test')
plt.plot(X_test, y_predict_3, label='forecast')
plt.title('Forecast vs Actuals')
plt.legend(loc='upper left', fontsize=18)
plt.show()


# In[208]:


rmse_hw = metrics.mean_squared_error(y_pred=y_predict_3,
                                       y_true=Y_test, squared = False)
print(rmse_hw)


# # ETS : RMSE=0.0764617977429886

# # 2.ARIMA：order=(7,0,19)

# In[188]:


kpss_stat, pval, lags, crit = kpss(cancellation, regression = 'c', nlags = 'auto')
print('p Value to the KPSS test is: ',pval)


# # (a) Apply Box-Cox to stabilize the variance.

# In[189]:


CancelRate, lmbda = boxcox(cancellation) 
print(CancelRate[:10])
print(lmbda)


# In[190]:


cancel_rate_BC=CancelRate
plt.figure(figsize = (20,8))
plt.plot(date_array, cancel_rate_BC)
plt.title('Daily cancel rate after Box-Cox transformation')


# # (b) Subtract the average for the corresponding day of the week to de-seasonality.

# In[191]:


# Generate day of the week information for each date in date_array
weekdays = [date.weekday() for date in date_array]  # Generate day of the week information, Monday is 0, Sunday is 6

for (date, is_canceled), count in grouped.items():
        filtered_0 = df[(df['date'] == date) & (df['is_canceled'] == 0)]
        filtered_1 = df[(df['date'] == date) & (df['is_canceled'] == 1)]
        count_0.append(filtered_0.shape[0])  
        count_1.append(filtered_1.shape[0])
        cancellation_rate=count_1[-1]/(count_1[-1]+count_0[-1])
        cancellation.append(cancellation_rate)    
        


# Calculate average cancellation rate by day of the week
avg_cancellation_by_weekday = {}
for weekday, cancellation_rate in zip(weekdays, cancellation):
    if weekday not in avg_cancellation_by_weekday:
        avg_cancellation_by_weekday[weekday] = []
    avg_cancellation_by_weekday[weekday].append(cancellation_rate)
avg_cancellation_by_weekday = {weekday: sum(rates) / len(rates) for weekday, rates in avg_cancellation_by_weekday.items()}

# Calculate deseasonalized cancellation rates
deseasonalized_cancellation_rates = []
for date, cancellation_rate in zip(date_array, cancellation):
    weekday = date.weekday()
    avg_rate = avg_cancellation_by_weekday[weekday]
    deseasonalized_rate = cancellation_rate - avg_rate
    deseasonalized_cancellation_rates.append(deseasonalized_rate)
    
    
deseasonalized_cancellation_rates 


# In[192]:


plt.figure(figsize = (20,8))
plt.plot(date_array, deseasonalized_cancellation_rates )
plt.title('Daily cancellation after Box-Cox transformation and de-seasonality')


# In[193]:


cancellation_BC_deseason = np.array(deseasonalized_cancellation_rates)
plot_pacf(cancellation_BC_deseason, title = 'PACF plot');


# ### p=7

# In[194]:


plot_acf(cancellation_BC_deseason, title = 'ACF plot');


# ### q=19

# In[209]:


model = ARIMA(Y_train, order=(7,0,19))
model_fit = model.fit()
y_predict = model_fit.forecast(14)


# In[211]:


plt.figure(figsize = (20,8))
plt.plot(X_train, Y_train, label = 'Train')
plt.plot(X_test, Y_test, label = 'Test')
plt.plot(X_test, y_predict, label='forecast')
plt.title('Forecast vs Actuals')
plt.legend(loc='upper left', fontsize=18)
plt.show()


# In[210]:


rmse_ses = metrics.mean_squared_error(y_pred=y_predict,y_true=Y_test, squared = False)
print(rmse_ses)


# # ARIMA : RMSE=0.07631304721880106

# # 2. seasonal ARIMA

# In[200]:


# plot the demand after differencing with lag = 7 
#The hysteresis is taken as 7 (one week)
cancellation_diff = cancellation_BC_deseason[7:] - cancellation_BC_deseason[:-7]
plt.figure(figsize = (20,8))
plt.plot(date_array[7:],cancellation_diff)
plt.show()


# The plot has no trend. So we do not take difference further, i.e., D = 0.

# In[201]:


plot_pacf(cancellation_diff);


# ### P=6

# In[202]:


plot_acf(cancellation_diff);


# ### Q=1

# In[203]:


#In the ARIMA model,order=(7,0,19), but because p and q need to be less than seasonal lag 7(Otherwise, the code reports an error), it is adjusted to this
model_2 = ARIMA(Y_train, order=(6,0,6), seasonal_order = (6,0,1,7))  
model_2_fit = model_2.fit()
y_predict_2 = model_2_fit.forecast(14)


# In[204]:


plt.figure(figsize = (20,8))
plt.plot(X_train, Y_train, label = 'Train')
plt.plot(X_test, Y_test, label = 'Test')
plt.plot(X_test, y_predict_2, label='forecast')
plt.title('Forecast vs Actuals')
plt.legend(loc='upper left', fontsize=18)
plt.show()


# In[213]:


rmse_ses = metrics.mean_squared_error(y_pred=y_predict_2,y_true=Y_test, squared = False)
print(rmse_ses)


# # seasonal ARIMA : RMSE=0.07150588215328124

# In[ ]:




