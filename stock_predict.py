import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas_datareader as data
from keras.models import load_model
import streamlit as st
from sklearn.preprocessing import MinMaxScaler
from datetime import date




start = '2015-01-01'
# end = '2021-12-31'
end = date.today().strftime("%Y-%m-%d")

st.title('Welcome to Stock Price Prediction of Your Favourite Company SAHIL')
# st.title('Stock Price Prediction')


user_input = st.text_input('Enter Stock Ticker', 'TTM')
df = data.DataReader(user_input, 'yahoo', start, end)

# Describing Data
st.subheader('Data from 2015 to till now')
st.write(df.describe())

#visualizations

st.subheader('Closing Price vs Time Chart')
chart = plt.figure(figsize = (10,5))
plt.plot(df.Close,  label = 'Closing Price')
st.pyplot(chart)

st.subheader('Closing Price vs Time Chart with 100MA')
ma100 = df.Close.rolling(100).mean()
chart = plt.figure(figsize = (10,5))
plt.plot(ma100,  label = 'MA100')
plt.plot(df.Close,  label = 'Closing Price')
st.pyplot(chart)

st.subheader('Closing Price vs Time Chart with 200MA')
ma100 = df.Close.rolling(100).mean()
ma200 = df.Close.rolling(200).mean()
chart = plt.figure(figsize = (10,5))
plt.plot(ma100, 'r',  label = 'MA100')
plt.plot(ma200, 'g',  label = 'MA200')
plt.plot(df.Close, 'b',  label = 'Closing Price')
st.pyplot(chart)

#splitting data into Trainng and Testing

data_training = pd.DataFrame(df['Close'][0:int(len(df)*0.70)])
data_testing =  pd.DataFrame(df['Close'][int(len(df)*0.70): int(len(df))])

scaler = MinMaxScaler(feature_range=(0,1))
data_training_array = scaler.fit_transform(data_training)


#Load my model into the keras file


model = load_model('kersa_model.h5')


#Testing part of the model


past_100_days_data = data_training.tail(100)
final_df = past_100_days_data.append(data_testing, ignore_index = True)
input_data = scaler.fit_transform(final_df)

x_test = []
y_test = []

for i in range(100, input_data.shape[0]):
    x_test.append(input_data[i-100: i])
    y_test.append(input_data[i, 0])


x_test, y_test = np.array(x_test), np.array(y_test)



# Making Predictions

y_predicted = model.predict(x_test)
scaler =  scaler.scale_

scale_factor = 1/ scaler[0]
y_predicted = y_predicted * scale_factor
y_test = y_test * scale_factor

#Final Graph
st.subheader('Prediction vs Original')
chart2 = plt.figure(figsize=(10,5))
plt.plot(y_test, 'r', label = 'Original Price')
plt.plot(y_predicted, 'b', label= 'Predicted Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(chart2)