import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import config as cf
from joblib import dump, load
from DataPreparation.main_preparation import data_processing

def RandomForest_fitting():
    data_processing(cf.raw_data, 29)
    # Get csv data
    data = pd.read_csv("../" + cf.prepared_data)
    X = data.drop(['0'], axis=1)
    y = data[['0']].values.ravel()


    # Splitting the dataset into the Training set and Test set
    X_Train, x_test, Y_Train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)


    # Feature Scaling
    StdScaler = StandardScaler()
    X_Train = StdScaler.fit_transform(X_Train)


    # Fitting the classifier into the Training set
    clf = RandomForestClassifier(n_estimators = 1000, min_samples_split=10, min_samples_leaf=1,
                                 max_features='sqrt', max_depth=70, bootstrap=False , random_state = 0)
    print('RandomForest fitting...')
    clf.fit(X_Train ,Y_Train)


    # Predicting the test set results
    pred = clf.predict(x_test)

    # Model Saving
    dump(clf, '../models/RandomForest_model.joblib')

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