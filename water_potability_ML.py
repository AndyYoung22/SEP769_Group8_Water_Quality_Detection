import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
#import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest
from sklearn import svm
from sklearn import metrics
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
import time
from sklearn.metrics import plot_confusion_matrix
from sklearn.utils import resample
import matplotlib.pyplot as plt 

#REF: https://stackoverflow.com/questions/53784971/how-to-disable-convergencewarning-using-sklearn by Omkar Sabade,Dec 15, 2018  

from warnings import filterwarnings
filterwarnings('ignore')

df= pd.read_csv("water_potability.csv")
df.info()

#Balance vs. imbalanced dataset
if_balance= pd.DataFrame(df["Potability"].value_counts())

#Handle missing values
df.isnull().sum()
df["ph"].fillna(value=df["ph"].mean(),inplace=True)
df["Sulfate"].fillna(value=df["Sulfate"].mean(),inplace=True)
df["Trihalomethanes"].fillna(value=df["Trihalomethanes"].mean(),inplace=True)

#REF: https://www.kaggle.com/code/smailaar/water-quality-ml by İSMAIL ACAR, Nov 19, 2022
#Check for outliers
for col_index, column in df.iteritems():
        if col_index == "index":
            continue
        #print(col_index)
        #print(df[col_index])
        col_mean, col_std = df[col_index].mean(), df[col_index].std()
        limitation = col_std * 3
        lower_limit, upper_limit = col_mean-limitation, col_mean+limitation
        #print(lower_limit, upper_limit)
        for row_index, row in df.iterrows():
            #print(df.at[row_index, col_index])
            if df.at[row_index, col_index] > upper_limit or df.at[row_index, col_index] < lower_limit:
                df.drop(row_index, axis=0, inplace=True)

#split first
yes = df[df['Potability']==1]
no = df[df['Potability']==0]
future_yes = resample(yes, replace = True, n_samples = 1900) #increase the number
#concate the new yes (small number of class) and no class
df = pd.concat([no,future_yes])

X=df.drop(columns=["Potability"]).values
Y=df["Potability"].values
X_train, X_val_test, Y_train, Y_val_test = train_test_split(X, Y, test_size = 0.2, random_state = 2)
X_val, X_test, Y_val, Y_test = train_test_split(X_val_test, Y_val_test, test_size = 0.5, random_state = 2)
#Normalize the data
scaler=StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val_test = scaler.fit_transform(X_val_test)
X_val = scaler.fit_transform(X_val)
X_test = scaler.fit_transform(X_test)


#SVM ############################################################################################
start_time = time.time()
svmModel = svm.LinearSVC().fit(X_train, Y_train)
print("training time is %s seconds" % (time.time() - start_time))
start_time1 = time.time()
Y_predict = svmModel.predict(X_val_test)
print("testing time is %s seconds" % (time.time() - start_time1))
acc = metrics.accuracy_score(Y_val_test, Y_predict)
print("The test accuracy for SVM ",acc)

cm_svm =  metrics.confusion_matrix(Y_val_test, Y_predict, normalize='all')
cmd_svm =  metrics.ConfusionMatrixDisplay(cm_svm)
print("The confusion matrix for SVM ")
print(cmd_svm)
cmd_svm.plot()
cmd_svm.ax_.set_title('SVM Confusion Matrix')
plt.show()


#Logistic ############################################################################################
#REF:https://towardsdatascience.com/logistic-regression-using-python-sklearn-numpy-mnist-handwriting-recognition-matplotlib-a6b31e2b166a
solver = "saga"
TrainNum, featureNum = X_train.shape
n_classes = 2
nums = range(1,11) #change epoch here
listnum = list(nums)

models = {
    "ovr": {"name": "One versus Rest",  "epochs": listnum},

}

accuray_list = []

for model in models:
    acc_best = 0
    epoch_best = 0
    Model_best = None
    m_paramters = models[model]
    for epoch in m_paramters ["epochs"]:
        lr = LogisticRegression(
            solver=solver,
            multi_class=model,
            penalty="l1",
            max_iter=epoch,
            random_state=3,
        )
        print("For epoch number",str(epoch))
        start_time = time.time()
        lr.fit(X_train, Y_train)
        print("training time is %s seconds" % (time.time() - start_time))
        start_time1 = time.time()
        Y_pred = lr.predict(X_val)
        print("testing time is %s seconds" % (time.time() - start_time1))
        
        accuracy_val = metrics.accuracy_score(Y_val, Y_pred)
        accuray_list.append(accuracy_val)
        print("For epoch",epoch,"the logistic accuracy is",accuracy_val)
        cm =  metrics.confusion_matrix(Y_val, Y_pred,normalize='all')
        print("The confusion matrix for Logistic Regression")
        print(cm)
        #REF:https://scikit-learn.org/stable/modules/generated/sklearn.metrics.ConfusionMatrixDisplay.html#sklearn.metrics.ConfusionMatrixDisplay
        cmd =  metrics.ConfusionMatrixDisplay(cm)
        cmd.plot()
        t = 'LR validation confusion matrix for epoch ' + str(epoch)
        cmd.ax_.set_title(t)
        plt.show()
      
        if accuracy_val>acc_best:
            acc_best = accuracy_val
            epoch_best = epoch
            Model_best = lr
    print("best validate accuracy:",acc_best)
    print("best epoch:",epoch_best)
    y_test_pred = Model_best.predict(X_test)
    accuracy_test = metrics.accuracy_score(Y_test, y_test_pred)
    print("accuracy for test",accuracy_test)
    cm_test =  metrics.confusion_matrix(Y_val, Y_pred,normalize='all')
    cmd_test =  metrics.ConfusionMatrixDisplay(cm_test)
    cmd_test.plot()
    cmd_test.ax_.set_title('LR test confusion matrix')
    plt.show()

               
plt.figure()
plt.plot(listnum,  accuray_list)
plt.xlabel("number of epochs")
plt.ylabel("logistic regression accuracy for validation ")
plt.show()
