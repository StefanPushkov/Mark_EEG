from joblib import dump, load
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.preprocessing import StandardScaler
import config as cf
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt

def prediction(data=cf.prepared_data):
    model = load('../models/RandomForest_model.joblib')
    data = pd.read_csv("../"+cf.prepared_data)
    data = data.loc[:100000]
    X = data.drop(['0'], axis=1)
    y = data[['0']]#.values.ravel()
    X = np.c_[X]
    # Feature Scaling
    StdScaler = StandardScaler()
    X_scaled = StdScaler.fit_transform(X)

    Y = label_binarize(y, classes=[0, 1, 2])
    n_classes = Y.shape[1]
    X_Train, x_test, Y_Train, y_test = train_test_split(X_scaled, Y, test_size=0.25, random_state=0)

    pred = model.predict_proba(x_test)

    # Plot the micro-averaged Precision-Recall curve
    # For each class
    precision = dict()
    recall = dict()
    average_precision = dict()
    for i in range(n_classes):
        precision[i], recall[i], _ = precision_recall_curve(y_test[:, i],
                                                            pred[:, i])
        average_precision[i] = average_precision_score(y_test[:, i], pred[:, i])

    # A "micro-average": quantifying score on all classes jointly
    precision["micro"], recall["micro"], _ = precision_recall_curve(y_test.ravel(),
                                                                    pred.ravel())
    average_precision["micro"] = average_precision_score(y_test, pred,
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
    plt.show()
    pred = model.predict(X)
    print('Accuracy metrics are evaluated')

    # Accuracy
    accu_percent = accuracy_score(y, pred) * 100
    print("Accuracy obtained over the whole training set is %0.6f %% ." % (accu_percent))

    # Balanced Accuracy Score
    blnc = balanced_accuracy_score(y, pred) * 100
    print("balanced_accuracy_score: %0.6f %% ." % (blnc))


prediction()
