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


    # PLOT FOR EACH CLASS
    from itertools import cycle
    # setup plot details
    colors = cycle(['navy', 'turquoise', 'darkorange', 'cornflowerblue', 'teal'])

    plt.figure(figsize=(7, 8))
    f_scores = np.linspace(0.2, 0.8, num=4)
    lines = []
    labels = []
    for f_score in f_scores:
        x = np.linspace(0.01, 1)
        y = f_score * x / (2 * x - f_score)
        l = plt.plot(x[y >= 0], y[y >= 0], color='gray', alpha=0.2)
        plt.annotate('f1={0:0.1f}'.format(f_score), xy=(0.9, y[45] + 0.02))

    lines.append(l)
    labels.append('iso-f1 curves')
    l = plt.plot(recall["micro"], precision["micro"], color='gold', lw=2)
    lines.append(l)
    labels.append('micro-average Precision-recall (area = {0:0.2f})'
                  ''.format(average_precision["micro"]))

    for i, color in zip(range(n_classes), colors):
        l, = plt.plot(recall[i], precision[i], color=color, lw=2)
        lines.append(l)
        labels.append('Precision-recall for class {0} (area = {1:0.2f})'
                      ''.format(i, average_precision[i]))

    fig = plt.gcf()
    fig.subplots_adjust(bottom=0.25)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Extension of Precision-Recall curve to multi-class')
    plt.legend(lines, labels, loc=(0, -.38), prop=dict(size=14))

    plt.show()

prediction()
