from turtle import color
from matplotlib.colors import Colormap
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 
import plotly.express as px


df = pd.read_csv(r'C:\Users\kelec\source\repos\House-Price-Predictor\Housing.csv')
print(df.head())
print(df.info())
print(df.describe())
print(df.columns)
print(df.isnull().sum())

#'price', 'area', 'bedrooms', 'bathrooms', 'stories', 'mainroad',
      # 'guestroom', 'basement', 'hotwaterheating', 'airconditioning',
      # 'parking', 'prefarea', 'furnishingstatus'],
    #  dtype='str'

#sns.pairplot(df, hue='price' , palette='viridis')
#plt.show()

#sns.scatterplot(x='price', y='area', data=df)
#plt.show()

#print(df.groupby('price').corr().sum)

df2 = df[['prefarea', 'furnishingstatus', 'mainroad',
          'guestroom', 'basement', 'hotwaterheating',
          'airconditioning']]
print(df2.head(20))

f_v = ['prefarea', 'furnishingstatus', 'mainroad',
       'guestroom', 'basement', 'hotwaterheating',
       'airconditioning']

def loop():
    for feat in f_v:
        sns.countplot(x=feat, data=df)
        plt.title(feat)
        plt.show()

loop()
# use dummy on funishingstatus
# check correlation of hotwaterheating with price

#sns.scatterplot(x='price', y='hotwaterheating', data = df)
#plt.show()


###### actual implement

df['prefarea'] = df['prefarea'].map({'yes': 1, 'no': 0})
df['mainroad'] = df['mainroad'].map({'yes': 1, 'no': 0})
df['guestroom'] = df['guestroom'].map({'yes': 1, 'no': 0})
df['basement'] = df['basement'].map({'yes': 1, 'no': 0})
df['hotwaterheating'] = df['hotwaterheating'].map({'yes': 1, 'no': 0})
df['airconditioning'] = df['airconditioning'].map({'yes': 1, 'no': 0})


df = pd.get_dummies(df, columns=['furnishingstatus'], drop_first=True, dtype=int)

"""df3= df[['prefarea','mainroad',
          'guestroom', 'basement', 'hotwaterheating',
          'airconditioning']]
          """
print(df.columns)
print(df.head(20))
print(df)
print(df.groupby('hotwaterheating')['price'].mean())

# data is cleaned

# standardize the values