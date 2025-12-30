import pandas as pd
import numpy as np
import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder,LabelEncoder,StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.metrics import confusion_matrix


dataset = pd.read_csv("Churn_Modelling.csv")

x = dataset.iloc[:, 3:-1].values
y = dataset.iloc[:,-1].values

lb = LabelEncoder()
x[:,2] = lb.fit_transform(x[:,2])

cl = ColumnTransformer([("Amin",OneHotEncoder(),[1])],remainder="passthrough")
x = cl.fit_transform(x)

x_train,x_test,y_train,y_test = train_test_split(x,y,random_state=0,test_size=0.25)

sc = StandardScaler()

x_train_scaled = sc.fit_transform(x_train)
x_test_scaled = sc.fit_transform(x_test)

ann = keras.Sequential()

ann.add(keras.layers.Dense(activation="relu", units=10))
ann.add(keras.layers.Dense(activation="relu", units=10))
ann.add(keras.layers.Dense(activation="sigmoid", units=1))

ann.compile(optimizer="adam",loss="binary_crossentropy", metrics=["accuracy"])
ann.fit(x_train_scaled,y_train, batch_size=16,epochs=20)

y_pred = ann.predict(x_test_scaled)
y_pred_labels = (y_pred>0.5).astype("int")
print(confusion_matrix(y_test,y_pred_labels))
