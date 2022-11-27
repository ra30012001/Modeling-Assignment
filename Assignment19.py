import pandas as pd
import numpy as np
import streamlit as st
from sklearn import linear_model, preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

heart = pd.read_csv("heart.csv")

# encode categorical variables
le = preprocessing.LabelEncoder()
for name in heart.columns:
    if heart[name].dtypes == 'O':
        heart[name] = heart[name].astype(str)
        le.fit(heart[name])
        heart[name] = le.transform(heart[name])

#Modeling
X = heart.loc[:, heart.columns != 'target']
y = heart['target']

#Train test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=1)

#Linear regression
reg = linear_model.LinearRegression()
reg.fit(X_train, y_train)

#predictions based on model
y_pred = reg.predict(X_test)

plt.scatter(y_pred, y_test, alpha = 0.7, color = 'b')
plt.xlabel('Predicted Disease')
plt.ylabel('Heart Desease')
plt.title('Linear Regression Model')

st.write(heart.head())
st.pyplot()
