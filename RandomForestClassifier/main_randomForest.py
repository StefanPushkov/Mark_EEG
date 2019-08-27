import time
from datetime import datetime
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, log_loss
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import config



def SVM_classifier():
    # Function to compute the classification using SVM
    def compute_Random_forest(X_train, y_train):

        clf = RandomForestClassifier(n_estimators=1000, min_samples_split=10, min_samples_leaf=1,
                                     max_features='sqrt', max_depth=70,
                                     bootstrap='False', random_state=0, n_jobs=-1)
        clf.fit(X_train, y_train)
        return clf

    # Function to calculate the accuracy
    def compute_accuracy(pred, y_test):
        # pred = c.predict(X_test)
        pred_accu = accuracy_score(y_test, pred)
        return pred_accu




    url = 'https://raw.githubusercontent.com/StefanPushkov/eeg_classification/master/converters/second_half.csv'
    data = config.data
    data = pd.read_csv('../'+data)
    data_tr = data.loc[:300000]
    #test_data = data.loc[500:74999]
    X = data_tr[['1', '2', '3', '4', '5', '6', '7', '8']]  # or data.drop(['0'], axis=1)
    y = data_tr[['0']]

    # Data Standardizing
    scaler_standard = StandardScaler()
    X_std_scaled = scaler_standard.fit_transform(X)

    # Data splitting
    X_train, X_test, y_train, y_test = train_test_split(X_std_scaled, y, test_size=0.5)  # when using .csv file
    start_time = time.time()
    print(" Started at ", datetime.utcfromtimestamp(int(time.time())).strftime('%Y-%m-%d %H:%M:%S'))
    model_clf = compute_Random_forest(X_train, y_train.values.ravel())


    # Save model to file
    filename = '../models/random_forest_model.sav'
    pickle.dump(model_clf, open(filename, 'wb'))


    print("Fitting time: %s seconds " % (time.time() - start_time))
    pred = model_clf.predict(X_test)

    # Accuracy metrics
    print('Accuracy metric is testing')
    accu_percent = compute_accuracy(pred, y_test.values.ravel()) * 100
    print("Accuracy obtained over the whole training set is %0.5f %% ." % (accu_percent))

    # AUC score metric
    fpr, tpr, thresholds = metrics.roc_curve(y_test, pred, pos_label=2)
    auc = metrics.auc(fpr, tpr)
    print("AUC: %0.5f %% ." % (auc))

    # Average precision score
    blnc = metrics.balanced_accuracy_score(y_test, pred)
    print("balanced_accuracy_score: %0.5f %% ." % (blnc))

    # Log_loss
    ll = log_loss(y_test, pred)
    print("Log-loss: %0.5f %% ." % (ll) * 100)

    '''
    # PLOTTING
    import matplotlib.pyplot as plt
    from sklearn.metrics import precision_recall_curve
    from sklearn.metrics import average_precision_score
    from sklearn.preprocessing import label_binarize

    y_score = pred
    Y = label_binarize(y, classes=[0, 1, 2])
    n_classes = Y.shape[1]

    # For each class
    precision = dict()
    recall = dict()
    average_precision = dict()
    for i in range(n_classes):
        precision[i], recall[i], _ = precision_recall_curve(y_test[:, i],
                                                            y_score[:, i])
        average_precision[i] = average_precision_score(y_test[:, i], y_score[:, i])

    precision["micro"], recall["micro"], _ = precision_recall_curve(y_test.ravel(),
                                                                    y_score.ravel())
    average_precision["micro"] = average_precision_score(y_test, y_score,
                                                         average="micro")
    print('Average precision score, micro-averaged over all classes: {0:0.2f}'
          .format(average_precision["micro"]))
    plt.figure()
    plt.step(recall['micro'], precision['micro'], color='b', alpha=0.2,
             where='post')
    plt.fill_between(recall["micro"], precision["micro"], alpha=0.2, color='b')

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title(
        'Average precision score, micro-averaged over all classes: AP={0:0.2f}'
            .format(average_precision["micro"]))
'''
SVM_classifier()

