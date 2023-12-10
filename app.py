import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 
import yfinance as yf
from keras.models import load_model
import streamlit as st
from datetime import date

start_date = "2000-01-01"
end_date = date.today()

st.title("Stock Trend Prediction")

user_input = st.text_input("Enter stock Ticker", "")
df = yf.download(user_input, start_date, end_date)
df.reset_index(inplace=True)

# Describing Data
st.subheader ("Data of 21st Century")
st.write(df.describe())

# Visualizations
st.subheader("Closing Price Vs Time Chart")
fig = plt.figure(figsize = (10, 10))
plt.plot(df.Close)
st.pyplot(fig)

st.subheader("100 MA-200 MA Closing Price Vs Time Chart")

ma100 = df.Close.rolling(100).mean()
ma200 = df.Close.rolling(200).mean()

fig = plt.figure(figsize = (12, 6))
plt.plot(ma100, 'r')
plt.plot(ma200, 'g')
plt.plot(df.Close, 'b')
st.pyplot(fig)

data_training = pd.DataFrame(df.Close[0:int(len(df) * 0.70)])
data_testing = pd.DataFrame(df.Close[int(len(df) * 0.70):int(len(df))])

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range = (0, 1))

data_training_array = scaler.fit_transform(data_training) 

x_train = []
y_train = []

for i in range(100, data_training_array.shape[0]):
    x_train.append(data_training_array[i-100:i])
    y_train.append(data_training_array[i, 0])

x_train, y_train = np.array(x_train), np.array(y_train)  

model = load_model("notebook\LSTM_StockPrice.h5")

past_100_days_test = data_training.tail(100)
data_testing_final = pd.concat([past_100_days_test, data_testing], ignore_index = True)

data_testing_final_scale = scaler.fit_transform(data_testing_final)

x_test = []
y_test = []

for i in range(100, data_testing_final_scale.shape[0]):
    x_test.append(data_testing_final_scale[i-100:i])
    y_test.append(data_testing_final_scale[i, 0])

x_test, y_test = np.array(x_test), np.array(y_test) 

y_predicted = model.predict(x_test)
scale_factor = 1/scaler.scale_
y_predicted = y_predicted * scale_factor
y_test = y_test * scale_factor

st.subheader("Predictions Vs Original")
fig2 = plt.figure(figsize = (10, 10))
plt.plot(y_test, 'r', label = "Original Price")
plt.plot(y_predicted, 'g', label = "Predicted Price")
plt.xlabel("Time")
plt.ylabel("Price")
plt.legend()
plt.show()
st.pyplot(fig2)