import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from joblib import dump, load
import config as cf
#import matplotlib.pyplot as graph




# Function to compute the classification using SVM
def compute_SVC(X_train, y_train):
    C = 1.0
    cache_size = 200
    class_weight = None
    coef0 = 0.0
    decision_function_shape = None
    degree = 3
    gamma = 'auto'
    kernel = 'rbf'
    max_iter = -1
    probability = False
    random_state = None
    shrinking = True
    tol = 0.001
    verbose = False
    c = svm.SVC(kernel='linear')
    c.fit(X_train, y_train)
    return c


# Function to calculate the accuracy
def compute_accuracy(X_test, y_test, c):
    pred = c.predict(X_test)
    print(pred)
    pred_accu = accuracy_score(y_test, pred)
    return pred_accu


# Function to compute the confusion matrix
def compute_confusion_matrix(test_f, test_l, c):
    pred = c.predict(test_f)
    x = confusion_matrix(test_l, pred)
    return x

# Function to split the data
def split_data(data_file, percent):
    tot = len(data_file)
    req_xt = int((float(percent)/100)*(tot))
    xt_get = []
    for s in range(0,(req_xt-1)):
        xt_get.append(data_file[s])
    yt_get = []
    for d in range(req_xt, tot):
        yt_get.append(data_file[d])
    return xt_get, yt_get

e
data = pd.read_csv("../"+cf.prepared_data)
print(data.head())
# data = data.loc[:75055]

X = data.drop(['0'], axis=1)
y = data[['0']].values.ravel()

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3) # when using .csv file


print('SVM is fitting......')
model_svc = compute_SVC(X_train, y_train)

print('Accuracy metric is testing')
accu_percent = compute_accuracy(X_test, y_test, model_svc) * 100
print("Accuracy obtained over the whole training set is %0.6f %% ." % (accu_percent))
#conf_mat = compute_confusion_matrix(features_train, labels_train, model_svc)
#print('Confusion matrix: ', conf_mat)
dump(model_svc, '../models/SVM_EEG.joblib')
