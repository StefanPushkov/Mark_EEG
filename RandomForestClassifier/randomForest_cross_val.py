import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler

import time
from datetime import datetime
import pickle

import matplotlib.pyplot as plt
from joblib import dump, load
import config as CONFIG

# DAta
url = 'https://raw.githubusercontent.com/StefanPushkov/eeg_classification/master/converters/second_half.csv'
data = pd.read_csv(url)
data = data.loc[25000:75000]

X = data[['2', '3', '4', '5', '6', '7']]  # or data.drop(['0'], axis=1)
y = data[['0']]


scaler_min_max = MinMaxScaler()
X_minmax_scaled = scaler_min_max.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_minmax_scaled, y, test_size=0.2)
y_train = y_train.values.ravel()



# Classifier
clf = RandomForestClassifier(random_state=0)



# Generate parameters
n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
max_features = ['auto', 'sqrt']
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
min_samples_split = [2, 5, 10]
min_samples_leaf = [1, 2, 4]
bootstrap = [True, False]

parameters_grid = {'n_estimators': n_estimators,
                   'max_features': max_features,
                   'max_depth': max_depth,
                   'min_samples_split': min_samples_split,
                   'min_samples_leaf': min_samples_leaf,
                   'bootstrap': bootstrap}




# Grid Search
clf_f = rf_random = RandomizedSearchCV(estimator = clf, param_distributions = parameters_grid,
                                       n_iter = 100, cv = 3, verbose=2, random_state=42)
start_time = time.time() # Time counter
print(" Started at ", datetime.utcfromtimestamp(int(time.time())).strftime('%Y-%m-%d %H:%M:%S'))
clf_f.fit(X_train, y_train)
print("Fitting time: %s seconds " % (time.time() - start_time))

print("Best score found on development set:")
print(clf_f.best_score_)

print("Best parameters set found on development set:")
print(clf_f.best_params_)

#print("Accuracy:"+str(np.average(cross_val_score(clf_f, X_train, y_train, scoring='accuracy', cv=3))))

