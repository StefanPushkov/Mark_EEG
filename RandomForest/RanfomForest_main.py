import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import config as cf
from joblib import dump, load
# from DataPreparation.main_preparation import data_processing

def RandomForest_fitting():
    # data_processing(cf.raw_data, 29)
    # Get csv data
    data = pd.read_csv(cf.prepared_data_15min)
    X = data.drop(['0'], axis=1)
    y = data[['0']].values.ravel()

    # Feature Scaling
    StdScaler = StandardScaler()
    X_scaled = StdScaler.fit_transform(X)

    # Splitting the dataset into the Training set and Test set
    X_Train, x_test, Y_Train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=0)



    # Fitting the classifier into the Training set
    clf = RandomForestClassifier(n_estimators=600, min_samples_split=2, min_samples_leaf=2,
                                 max_features='auto', max_depth=60, bootstrap=False, random_state=0, verbose=10, n_jobs=8)

    print('RandomForest fitting...')
    clf.fit(X_Train, Y_Train)

    # Predicting the test set results
    pred = clf.predict(x_test)

    # Model Saving
    dump(clf, '../models/RandomForest_model_15min.joblib')

    # Testing accuracy
    print('Accuracy metrics are evaluated')

    # Accuracy
    accu_percent = accuracy_score(y_test, pred) * 100
    print("Accuracy obtained over the whole training set is %0.6f %% ." % (accu_percent))

    # Balanced Accuracy Score
    blnc = balanced_accuracy_score(y_test, pred) * 100
    print("balanced_accuracy_score: %0.6f %% ." % (blnc))

if __name__ == "__main__":
    RandomForest_fitting()