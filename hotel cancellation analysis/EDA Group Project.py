#!/usr/bin/env python
# coding: utf-8

# # Group discussion

# In[20]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load the dataset
data = pd.read_csv("hotel_booking_data_cleaned.csv")


# In[21]:


data


# In[22]:


data.info()


# In[23]:


data.describe()


# In[24]:


data.isna().sum()


# In[25]:


data['distribution_channel'].value_counts()


# In[26]:


data['meal'].value_counts()


# In[27]:


data['market_segment'].value_counts()


# In[33]:


# 删除 adr<0 和 adr>5000 的行  
hotel = hotel[(hotel['adr'] >= 0) & (hotel['adr'] <= 5000)]  
#替换缺失项得到新数据
nan_replacements = {"children": 0,"country": "Unknown", "agent": 0, "company": 0}
hotel_cln = data.fillna(nan_replacements)
#替换full_data_cln中不规范值
#meal字段包含'Undefined'意味着自带食物SC
#关于meal字段缩写代表的意义，########333
hotel_cln["meal"].replace("Undefined", "SC", inplace=True)
hotel_cln["market_segment"].replace("Undefined", "Complementary", inplace=True)
hotel_cln["distribution_channel"].replace("Undefined", "Direct", inplace=True)
print(hotel_cln)


# In[34]:


hotel_cln.isnull().sum()


# In[35]:


from sklearn.preprocessing import LabelEncoder
categorical_columns = hotel_cln.select_dtypes(include=['object']).columns
label_encoder = LabelEncoder()
for column in categorical_columns:
    hotel_cln[column]= label_encoder.fit_transform(hotel_cln[column])

print (hotel_cln)


# ## Customer_type

# Type of booking, assuming one of four categories:
# 
# Contract - when the booking has an allotment or other type of contract associated to it;
# 合同 - 当预订与配额或其他类型的合同相关联时;
# 
# Group – when the booking is associated to a group;
# 团体 - 当预订与团体相关联时;
# 
# Transient – when the booking is not part of a group or contract, and is not associated to other transient booking;
# 短期 - 当预订不是团体或合同的一部分，并且不与其他短期预订相关时；
# Transient travellers can include:
# 
# Walk-in guests
# Guests with a last-minute booking, and/or
# Simply individual guests requiring a short stay at the hotel
# 
# Transient-party – when the booking is transient, but is associated to at least other transient booking：
# 短暂团体 - "Transient-party" 是一个术语，通常用于酒店和旅游行业，指的是临时性团体或聚会。这种类型的团体通常是临时性的，不像传统的固定团体或会议，而是由个人或小团体组成的短期聚会或活动

# In[3]:


Transient = data[data['customer_type'] == 'Transient']
Transient['is_canceled'].value_counts(normalize = True)


# In[4]:


Contract = data[data['customer_type'] == 'Contract']
Contract['is_canceled'].value_counts(normalize = True)


# In[5]:


Party = data[data['customer_type'] == 'Transient-Party']
Party['is_canceled'].value_counts(normalize = True)


# In[6]:


Group = data[data['customer_type'] == 'Group']
Group['is_canceled'].value_counts(normalize = True)


# 对于顾客群体来说，其中大部分都是临时顾客，而在临时顾客群体中取消率高达40.7%，酒店可以针对临时顾客实行更有力的惩罚机制来降低临时顾客的取消率；
# 与此同时，与酒店事先签有合同的群体同样有高达30.9%的取消率，酒店应当重新审视合同内容与违约金的设置。Maybe 可以联系临时顾客是否需要缴纳deposit来进一步说明。

# In[7]:


plt.figure(figsize=(12, 6))
sns.countplot(x='customer_type', hue='is_canceled', data=data)
plt.title('Cancellation Distribution by customer_type')
plt.xticks(rotation=45)
plt.show()


# In[8]:


customertype_data = data[["is_canceled","customer_type"]]
#y表示取消订单，n表示未取消订单
customertype_data_y = customertype_data[customertype_data["is_canceled"]==1].groupby("customer_type")["customer_type"].count()
customertype_data_n = customertype_data[customertype_data["is_canceled"]==0].groupby("customer_type")["customer_type"].count()
customertype_data_y


# In[9]:


plt.pie(x=customertype_data_y.values,labels=customertype_data_y.index,autopct="%.1f%%")
plt.title("canceled customer_type")


# ## total_of_special_requests：Number of special requests made by the customer (e.g. twin bed or high floor)

# In[10]:


plt.figure(figsize=(10, 6))
sns.barplot(x='total_of_special_requests', y='is_canceled', data=data)
plt.title('Cancellation Rate Based on Number of Special Requests')
plt.show()


# In[11]:


plt.figure(figsize=(12, 6))
sns.countplot(x='total_of_special_requests', hue='is_canceled', data=data)
plt.title('Cancellation Distribution by total_of_special_requests')
plt.xticks(rotation=45)
plt.show()


# In[12]:


plt.figure(figsize=(10, 6))
sns.lineplot(x='total_of_special_requests', y='is_canceled', data=data)
plt.title('Cancellation Rate Based on Number of Special Requests')
plt.show()


# 很明显，随着特殊要求的增多，顾客的取消率呈逐步下跌的趋势，那么对与酒店来说，要尽可能的去满足顾客的需求，包括但不限于完善基础设施等等（后续maybe可以详细地去了解一下欧洲人在订酒店时具体会有哪些要求来具体问题具体分析。）而为了留住客人，对于那些要求较高的用户，酒店也可以推出类似于常客计划或是会员制，将一些进阶服务需求加入会员专属特权，win-win。
# A specific room according to your preferences
# A city map
# Discounted room rates
# Phone chargers or adapters
# Umbrellas
# Netflix or other on-demand entertainment
# Bicycles
# A nightlight
# Room amenities, including pillows, bathroom toiletries, slippers, and clean linens
# Turndown service
# Extra Breakfast Options
# Treats for Special Occasions.

# ## previous_cancellations

# In[13]:


#研究之前改变的次数对取消预订率的影响
previous_cancellations = list(
    data.groupby('previous_cancellations').size().sort_values(ascending=False).head(15).index)
data[data.previous_cancellations.isin(previous_cancellations)].shape[0] / data.shape[0]

fig, ax1 = plt.subplots(figsize=(10, 6))
ax2 = ax1.twinx()
plt.xticks(range(len(previous_cancellations)),previous_cancellations)
ax1.bar(
    range(len(previous_cancellations)), data[data.previous_cancellations.isin(previous_cancellations)].groupby('previous_cancellations').size().sort_values(ascending=False))
ax1.set_xlabel('previous_cancellations')
ax1.set_ylabel('Counts')
ax2.plot(
    range(len(previous_cancellations)),
    data[data.previous_cancellations.isin(previous_cancellations)].groupby('previous_cancellations')['is_canceled'].mean().loc[previous_cancellations], 'ro-')
ax2.set_ylabel('Cancellation rate')


# 绝大部分的旅客都是之前没有取消过的旅客，但是还是有几个指标是值得注意的，在之前取消过一次的顾客中，取消率极高；对于取消次数超过6次的顾客中再次取消的概率也是非常高，那么对于先前有取消过的顾客，酒店方面要做好及时的客服沟通工作，比如电话回访等等，询问顾客取消的原因，而对于先前取消次数过高的顾客，酒店需要采取黑名单的措施，防止类似的情况再次发生。

# ## is_repeated_guest

# In[14]:


plt.figure(figsize=(12, 6))
sns.countplot(x='is_repeated_guest', hue='is_canceled', data=data)
plt.title('Cancellation Distribution by customer_type')
plt.xticks(rotation=45)
plt.show()


# In[15]:


data['is_repeated_guest']=data['is_repeated_guest'].astype('str').replace(['0','1'],['no','yes'])


# In[16]:


Yes = data[data['is_repeated_guest'] == 'yes']
Yes['is_canceled'].value_counts(normalize = True)


# 回头客群体虽然人数较少，但是整体的取消率只要14%，酒店可以通过会员制等措施来增加客户粘性。

# # Modeling

# ## correlation heatmap

# In[21]:


numerical_corr = data.select_dtypes(include=[np.number]).corr()
plt.figure(figsize=(12, 10))
sns.heatmap(numerical_corr, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Heatmap for Numerical Features')
plt.show()


# In[25]:


# 使用独热编码将分类变量转换为虚拟变量
data_encoded = pd.get_dummies(data)

# 计算转换后数据集中变量之间的相关性
correlation_matrix = data_encoded.corr()

# 打印相关性矩阵
print(correlation_matrix)


# In[ ]:




