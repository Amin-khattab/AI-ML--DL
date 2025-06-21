import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd
from sklearn.preprocessing import LabelEncoder,OneHotEncoder,StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
import keras as keras
from sklearn.metrics import confusion_matrix

dataset = pd.read_csv("Churn_Modelling.csv")
x = dataset.iloc[:,3:-1].values
y = dataset.iloc[:,-1].values


encoder = LabelEncoder()
x[:,2] = encoder.fit_transform(x[:,2])

cl = ColumnTransformer(transformers=[("amin",OneHotEncoder(),[1])],remainder="passthrough")
x = cl.fit_transform(x)

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.25,random_state=0)

sc = StandardScaler()

x_train_scaled = sc.fit_transform(x_train)
x_test_scaled = sc.transform(x_test)

ann = keras.models.Sequential()
ann.add(keras.layers.Dense(activation="relu",units=6))
ann.add(keras.layers.Dense(activation="relu",units=6))
ann.add(keras.layers.Dense(activation="sigmoid",units=1))

ann.compile(optimizer='adam',metrics=['accuracy'],loss='binary_crossentropy')

ann.fit(x_train_scaled,y_train,batch_size=1000,epochs=100)

y_pred = ann.predict(x_test_scaled)
y_pred_labels = (y_pred > 0.5).astype(int)
print(confusion_matrix(y_test, y_pred_labels))

