import pandas as pd
import numpy as py
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno

df= pd.read_csv("water_potability.csv")
df.info()
#plot the distribution of data
list = []
k=1
df1=df.describe().T
for name in df.columns:
    unique_values = len(df[name].unique())
    if unique_values > 2:
        list.append(name)
plt.figure(figsize=(16,16))
for i in df.loc[:,list]:
    plt.subplot(3,3,k)
    plt.title(i)
    sns.histplot(df[i])
    k+=1
plt.show()
#distribution of the feature “Potability"
if_balance=pd.DataFrame(df["Potability"].value_counts())
if_balance.plot.pie(labels=['Not Potable','Potable'],subplots=True,autopct='%.2f',figsize=(10,5))
plt.show()

#REF:https://medium.com/towards-data-science/seaborn-heatmap-for-visualising-data-correlations-66cbef09c1fe
#correlation heatmap
heatmap=sns.heatmap(df.corr())
plt.show()

#REF:https://towardsdatascience.com/using-the-missingno-python-library-to-identify-and-visualise-missing-data-prior-to-machine-learning-34c8c5b5f009
#plot the missing values
df2=df.isnull().sum()
msno.matrix(df)
plt.show()
