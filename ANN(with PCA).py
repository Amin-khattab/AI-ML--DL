import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import keras
from sklearn.metrics import mean_squared_error,r2_score

dataset = pd.read_csv("complex_regression_data.csv")
x = dataset.iloc[:,:-1].values
y = dataset.iloc[:,-1].values

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=0)

sc_x = StandardScaler()

x_train_scaled = sc_x.fit_transform(x_train)
x_test_scaled = sc_x.transform(x_test)

DR = PCA(n_components=0.95)
DR_x_train = DR.fit_transform(x_train_scaled)
DR_x_test = DR.transform(x_test_scaled)

ann = keras.Sequential()

ann.add(keras.layers.Dense(units=10,activation=None))
ann.add(keras.layers.BatchNormalization())
ann.add(keras.layers.Activation("relu"))
ann.add(keras.layers.Dropout(0.2))

ann.add(keras.layers.Dense(units=10,activation=None))
ann.add(keras.layers.BatchNormalization())
ann.add(keras.layers.Activation("relu"))
ann.add(keras.layers.Dropout(0.2))

ann.add(keras.layers.Dense(units=10,activation=None))
ann.add(keras.layers.BatchNormalization())
ann.add(keras.layers.Activation("relu"))
ann.add(keras.layers.Dropout(0.2))

ann.add(keras.layers.Dense(units=1))

ann.compile(optimizer="adam",loss="mean_squared_error")
ann.fit(DR_x_train,y_train,epochs=100,batch_size=32)

y_pred = ann.predict(DR_x_test)

print(r2_score(y_test,y_pred))
print(mean_squared_error(y_test,y_pred))
