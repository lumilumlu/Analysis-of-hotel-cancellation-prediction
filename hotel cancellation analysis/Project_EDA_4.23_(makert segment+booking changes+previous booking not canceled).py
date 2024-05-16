#!/usr/bin/env python
# coding: utf-8

# In[54]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data=pd.read_csv('hotel_booking_data_cleaned.csv',usecols=['is_canceled','market_segment','booking_changes','previous_bookings_not_canceled','distribution_channel']) 


# # 1.market_segment  &  cancellation

# In[16]:


data['market_segment'].value_counts()


# In[3]:


Direct = data[data['market_segment']=='Direct']
Direct['is_canceled'].value_counts(normalize=True)


# In[4]:


Corporate = data[data['market_segment']=='Corporate']
Corporate['is_canceled'].value_counts(normalize=True)


# In[5]:


Online_TA = data[data['market_segment']=='Online TA']
Online_TA['is_canceled'].value_counts(normalize=True)


# In[7]:


Offline_TATO = data[data['market_segment']=='Offline TA/TO']
Offline_TATO['is_canceled'].value_counts(normalize=True)


# In[8]:


Complementary = data[data['market_segment']=='Complementary']
Complementary['is_canceled'].value_counts(normalize=True)


# In[9]:


Groups = data[data['market_segment']=='Groups']
Groups['is_canceled'].value_counts(normalize=True)


# In[10]:


Aviation = data[data['market_segment']=='Aviation']
Aviation['is_canceled'].value_counts(normalize=True)


# In[17]:


Undefined = data[data['market_segment']=='Undefined']
Undefined['is_canceled'].value_counts(normalize=True)


# In[46]:


cross_table = pd.crosstab(index=data['market_segment'], columns=data['is_canceled'], normalize='index')

ax = cross_table.plot(kind='bar', stacked=True)


height_cumulative = [0] * len(cross_table)  

for i, bar in enumerate(ax.patches):
    bar_width = bar.get_width()
    bar_x = bar.get_x()
    bar_height = bar.get_height()
    
    text_y = height_cumulative[i % len(cross_table)] + bar_height / 2

    height_cumulative[i % len(cross_table)] += bar_height

    ax.annotate(f'{bar_height:.2f}', 
                (bar_x + bar_width / 2, text_y), 
                ha='center', va='center', xytext=(0, 5), 
                textcoords='offset points')

plt.xlabel('Market Segment')
plt.ylabel('Cancellation Rate')
plt.legend(title='Canceled', loc='upper right', labels=['Not Canceled', 'Canceled'])
plt.show()


# # 2.distribution_channel  &  cancellation

# In[26]:


data['distribution_channel'].value_counts()


# In[27]:


ds_TATO = data[data['distribution_channel']=='TA/TO']
ds_TATO['is_canceled'].value_counts(normalize=True)


# In[28]:


dc_Direct = data[data['distribution_channel']=='Direct']
dc_Direct['is_canceled'].value_counts(normalize=True)


# In[29]:


dc_Corporate = data[data['distribution_channel']=='Corporate']
dc_Corporate['is_canceled'].value_counts(normalize=True)


# In[32]:


dc_GDS = data[data['distribution_channel']=='GDS']
dc_GDS['is_canceled'].value_counts(normalize=True)


# In[33]:


dc_Undefined = data[data['distribution_channel']=='Undefined']
dc_Undefined['is_canceled'].value_counts(normalize=True)


# In[48]:


cross_table = pd.crosstab(index=data['distribution_channel'], columns=data['is_canceled'], normalize='index')  
ax = cross_table.plot(kind='bar', stacked=True)


height_cumulative = [0] * len(cross_table)  

for i, bar in enumerate(ax.patches):
    bar_width = bar.get_width()
    bar_x = bar.get_x()
    bar_height = bar.get_height()
    
    text_y = height_cumulative[i % len(cross_table)] + bar_height / 2

    height_cumulative[i % len(cross_table)] += bar_height

    ax.annotate(f'{bar_height:.2f}', 
                (bar_x + bar_width / 2, text_y), 
                ha='center', va='center', xytext=(0, 5), 
                textcoords='offset points')
plt.xlabel('Distribution channel')  
plt.ylabel('Cancellation')  
plt.legend(title='canceled', loc='upper right', labels=['Not Canceled', 'Canceled'])  
plt.show()  


# # 3.previous_bookings_not_canceled  & booking_changes & cancellation

# In[53]:


groups = data.groupby("is_canceled")
for name, group in groups:
    plt.scatter(group['previous_bookings_not_canceled'],group['booking_changes'], marker="o", label=name)
plt.xlabel('previous_bookings_not_canceled')  
plt.ylabel('booking_changes')  
plt.legend(title='canceled', loc='upper right', labels=['Not Canceled', 'Canceled'])  
plt.show()


# In[ ]:




