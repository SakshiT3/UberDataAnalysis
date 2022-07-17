#!/usr/bin/env python
# coding: utf-8

# In[40]:


import pandas as pd
from sklearn.linear_model import LogisticRegression,LinearRegression
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import itertools
import gc
import os
import sys
get_ipython().run_line_magic('matplotlib', 'inline')


# In[41]:


# Memory saving function credit to https://www.kaggle.com/gemartin/load-data-reduce-memory-usage
def reduce_mem_usage(df):
    """ iterate through all the columns of a dataframe and modify the data type
        to reduce memory usage.
    """
    start_mem = df.memory_usage().sum() / 1024**2
    
    for col in df.columns:
        col_type = df[col].dtype
        
        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                #if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                #    df[col] = df[col].astype(np.float16)
                #el
                if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        #else:
            #df[col] = df[col].astype('category')

    end_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage of dataframe is {:.2f} MB --> {:.2f} MB (Decreased by {:.1f}%)'.format(
        start_mem, end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df


# In[42]:


cab_data = pd.read_csv("cab_rides.csv")
cab_data=reduce_mem_usage(cab_data)
weather_data = pd.read_csv("weather.csv")
weather_data=reduce_mem_usage(weather_data)


# In[43]:


cab_data


# In[46]:


import datetime
cab_data['datetime']= pd.to_datetime(cab_data['time_stamp'])
cab_data
weather_data['date_time'] = pd.to_datetime(weather_data['time_stamp1'])


# In[47]:


weather_data.columns


# In[48]:


cab_data.shape


# In[49]:


weather_data.shape


# In[50]:


cab_data.describe()


# In[51]:


weather_data.describe()


# In[52]:


a=pd.concat([cab_data,weather_data])
a['day']=a.date_time.dt.day
a['hour']=a.date_time.dt.hour
a.fillna(0,inplace=True)
a.columns


# In[53]:


a.groupby('cab_type').count()


# In[54]:


a.groupby('cab_type').count().plot.bar()


# In[55]:


a['price'].value_counts().plot(kind='bar',figsize=(100,50),color='blue')


# In[56]:


a['hour'].value_counts().plot(kind='bar',figsize=(10,5),color='blue')


# In[57]:


import matplotlib.pyplot as plt
x=a['hour']
y=a['price']
plt.plot(x,y)
plt.show()


# In[58]:


x=a['rain']
y=a['price']
plt.plot(x,y)
plt.show()


# In[59]:


a.columns


# In[60]:


x1=a[['distance', 'temp','clouds', 'pressure', 'humidity','wind','rain','day','hour','surge_multiplier','clouds']]
y1=a['price']


# In[61]:


# Using Skicit-learn to split data into training and testing sets
from sklearn.model_selection import train_test_split
# Split the data into training and testing sets
x_train, y_train, x_test, y_test = train_test_split(x1, y1, test_size = 0.25, random_state = 42)


# In[62]:


linear=LinearRegression()
linear.fit(x_train,x_test)
LinearRegression(copy_X=True, fit_intercept=True, n_jobs=None, normalize=False)
predictions=linear.predict(y_train)


# In[63]:


df = pd.DataFrame({'Actual': y_test, 'Predicted': predictions})
df


# In[64]:


df1 = df.head(25)
df1.plot(kind='bar',figsize=(26,10))
plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
plt.show()


# In[65]:


import pickle
pickle_out = open("linear.pkl","wb")
pickle.dump(linear, pickle_out)
pickle_out.close() 


# In[66]:


import streamlit as st

pickle_in = open('linear.pkl','rb')
linear = pickle.load(pickle_in)

def predict_ride_fair(distance,cab_type,time_stamp,destination,source,price,surge_multiplier,
                                 id,product_id,name,temp,location,clouds,pressure,rain,time_stamp1,humidity,wind):
    prediction = linear.predict([[distance,cab_type,time_stamp,destination,source,price,surge_multiplier,
                                 id,product_id,name,temp,location,clouds,pressure,rain,time_stamp1,humidity,wind]])
    print(prediction)
    return Prediction

def main():
    st.title("Uber Ride Fair Predictor")
    html_temp = """
    <div style="background-color:tomato;padding:10px">
    <h2 style="color:white;text-align:center;">streamlit Uber Ride Predictor ML App </h2>
    </div>
    """
    st.markdown(html_temp,unsafe_allow_html=True)
    distance = st.text_input("distance","Type Here")
    cab_type = st.text_input("cab_type","Type Here")
    time_stamp = st.text_input("time_stamp","Type Here")
    destination = st.text_input("destination","Type Here")
    source = st.text_input("source","Type Here")
    price = st.text_input("price","Type Here")
    surge_multiplier = st.text_input("surge_multiplier","Type Here")
    id = st.text_input("id","Type Here")
    product_id = st.text_input("product_id","Type Here")
    name = st.text_input("name","Type Here")
    
    temp = st.text_input("temp","Type Here")
    location = st.text_input("location","Type Here")
    clouds = st.text_input("clouds","Type Here")
    pressure = st.text_input("pressure","Type Here")
    rain = st.text_input("rain","Type Here")
    time_stamp1 = st.text_input("time_stamp1","Type Here")
    humidity = st.text_input("humidity","Type Here")
    wind = st.text_input("wind","Type Here")
    
    result =""
    if st.button("Predict"):
        result = predict_ride_fair(distance,cab_type,time_stamp,destination,source,price,surge_multiplier,
                                 id,product_id,name,temp,location,clouds,pressure,rain,time_stamp1,humidity,wind)
    st.success('The output is {}'.format(result))
    if st.button('About'):
        st.text("Less Learn")
        st.text("Built with Streamlit")
        

if __name__=="__main__":
    main()


# In[ ]:




