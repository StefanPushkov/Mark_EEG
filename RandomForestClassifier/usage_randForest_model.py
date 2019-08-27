import pickle
import pandas as pd
from sklearn import metrics
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn import metrics
from sklearn.metrics import log_loss
from sklearn.metrics import accuracy_score
import config




# 'DataPreparation/final_merged.csv'
c_data = config.data
data = pd.read_csv('../'+c_data)
#data_tr = data.loc[155001:]
test_data = data.loc[300001:]
X = test_data[['1', '2', '3', '4', '5', '6', '7', '8']]  # or data.drop(['0'], axis=1)
y = test_data[['0']]

# Data Standardizing
scaler_standard = StandardScaler()
X_std_scaled = scaler_standard.fit_transform(X)

# Data splitting
#X_train, X_test, y_train, y_test = train_test_split(X_std_scaled, y, test_size=0.2)

with open('../models/random_forest_model.sav', 'rb') as f:
    clf = pickle.load(f)

def prediction(X_features, clf=clf):

    # Scaling features before prediction using either StandardScaler or MinMaxScaler
    sc = StandardScaler()
    # min_max = MinMaxScaler()
    X_std_scaled = sc.fit_transform(X_features)
    print('Prediction...')
    predicted_class = clf.predict(X_std_scaled)

    # Accuracy score metric
    accu_percent = accuracy_score(y.values.ravel(), predicted_class) * 100
    print("Accuracy obtained over the whole training set is %0.5f %% ." % (accu_percent))

    # AUC score metric
    fpr, tpr, thresholds = metrics.roc_curve(y, predicted_class, pos_label=2)
    auc = metrics.auc(fpr, tpr)
    print("AUC: %0.5f %% ." % (auc))

    # Average precision score
    blnc = metrics.balanced_accuracy_score(y, predicted_class)
    print("balanced_accuracy_score: %0.5f %% ." % (blnc))

    # Log-loss metric
    ll = log_loss(y, predicted_class)
    print("Log-loss: %0.5f %% ." % (ll) * 100)
    '''
    # PLOTTING

    # Data Standardizing
    scaler_standard = StandardScaler()
    X_std_scaled = scaler_standard.fit_transform(X)

    # Data splitting
    X_train, X_test, y_train, y_test = train_test_split(X_std_scaled, y, test_size=0.2)
    import matplotlib.pyplot as plt
    from sklearn.metrics import precision_recall_curve
    from sklearn.metrics import average_precision_score
    from sklearn.preprocessing import label_binarize

    y_score = predicted_class
    Y = label_binarize(y, classes=[0, 1, 2])
    n_classes = Y.shape[1]

    # For each class
    precision = dict()
    recall = dict()
    average_precision = dict()
    for i in range(n_classes):
        precision[i], recall[i], _ = precision_recall_curve(y_test.iloc[:, i],
                                                            y_score[i])
        average_precision[i] = average_precision_score(y_test.iloc[:, i], y_score[i])

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
    plt.title('Average precision score, micro-averaged over all classes: AP={0:0.2f}'.format(average_precision["micro"]))
    '''




prediction(X_std_scaled)