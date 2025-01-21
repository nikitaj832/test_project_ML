import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score

#upload data
df =pd.read_csv("C:\\Users\\KMC\\Desktop\\pandas\\insurance.csv")
df.head(3)
df.isnull().sum()

#labelEncoder
lb = LabelEncoder()

df['sex'] =lb.fit_transform(df['sex'])
df['region'] =lb.fit_transform(df['region'])
df['smoker'] =lb.fit_transform(df['smoker'])

#train_test_split
x= df.drop(columns =['charges'])
y= df['charges']

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size =0.2,random_state =42)

#LinearRegression
lr =LinearRegression()

lr.fit(x_train,y_train)
y_pred =lr.predict(x_test)

#r2_score
print("r2_score :",r2_score(y_test,y_pred))