import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense,Dropout,LSTM
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from sklearn.naive_bayes import MultinomialNB

data=pd.read_csv("/content/drive/My Drive/Google-Stock-Price-Prediction-Using-RNN---LSTM-master/GOOG.csv")

data.head()

data_train=data[data["Date"]<"2019-01-01"].copy()
data_train.head()

data_test=data[data["Date"]>="2019-01-01"].copy()
data_test.head()

training_data=data_train.drop(["Date","Adj Close"],axis=1)
training_data.head()

testing_data=data_test.drop(["Date","Adj Close"],axis=1)
testing_data.head()

scalar=MinMaxScaler()
training_data=scalar.fit_transform(training_data)
training_data.shape

testing_data=scalar.fit_transform(testing_data)
testing_data.shape

x_train=[]
y_train=[]

for i in range(60,training_data.shape[0]):
  x_train.append(training_data[i-60:i])
  y_train.append(training_data[i,0])

x_train,y_train=np.array(x_train),np.array(y_train)

x_train.shape,y_train.shape

model=Sequential()
model.add(LSTM(units=50,activation='relu',return_sequences=True,input_shape=(x_train.shape[1],5)))
model.add(Dropout(0.2))

model.add(LSTM(units=60,activation='relu',return_sequences=True))
model.add(Dropout(0.3))

model.add(LSTM(units=80,activation='relu',return_sequences=True))
model.add(Dropout(0.4))

model.add(LSTM(units=120,activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(units=1))
model.summary()

model.compile(optimizer='adam',loss='mean_squared_error')

model.fit(x_train,y_train,epochs=5,batch_size=1000)

x_train.shape

past_60_days=data_train.tail(60)

df=past_60_days.append(data_test,ignore_index=True)

df=df.drop(['Date','Adj Close'],axis=1)

inputs=scalar.transform(df)

x_test=[]
y_test=[]

for i in range(60,inputs.shape[0]):
  x_test.append(inputs[i-60:i])
  y_test.append(inputs[i,0])

x_test,y_test=np.array(x_test),np.array(y_test)

x_test.shape,y_test.shape

y_pred=model.predict(x_test)

y_test=y_test*(1/scalar.scale_[0])
y_pred=y_pred*(1/scalar.scale_[0])
y_test=y_test.reshape(y_test.shape[0],1)
y_pred=y_pred.reshape(y_pred.shape[0],1)

y_test.shape

plt.plot(y_pred,color="red",label="Predicted")
plt.plot(y_test,color="green",label="Actual")
plt.legend()
plt.show()