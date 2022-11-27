import pandas as pd
import numpy as np
import streamlit as st
import seaborn as sns
from scipy.stats import skew
from sklearn import datasets, linear_model, metrics, preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

df = pd.read_csv("heart.csv")

#Modeling
X = df.loc[:, df.columns != 'target']
y = df['target']

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