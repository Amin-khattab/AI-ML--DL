import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd
import keras as keras
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error  

dataset = pd.read_excel("Folds5x2_pp.xlsx")
x = dataset.iloc[:,:-1].values
y = dataset.iloc[:,-1].values.reshape(-1,1)

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=0)

sc_x =StandardScaler()
sc_y = StandardScaler()

x_train_scaled = sc_x.fit_transform(x_train)
x_test_scaled = sc_x.transform(x_test)
y_train_scaled = sc_y.fit_transform(y_train)

ann = keras.models.Sequential()

ann.add(keras.layers.Dense(activation="relu",units=6))
ann.add(keras.layers.Dense(activation="relu",units=6))
ann.add(keras.layers.Dense(units=1))

ann.compile(optimizer="adam",loss="mean_squared_error")

ann.fit(x_train_scaled,y_train_scaled,epochs=100,batch_size=100)

y_pred = ann.predict(x_test_scaled)

z = np.column_stack((y_test,sc_y.inverse_transform(y_pred)))
print(z)


print("MSE:", mean_squared_error(y_test, sc_y.inverse_transform(y_pred)))