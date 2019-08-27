from joblib import dump, load
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.preprocessing import StandardScaler
import config as cf

def prediction(data=cf.prepared_data):
    model = load('../models/RandomForest_model.joblib')
    data = pd.read_csv(data)
    data = data.loc[:]
    X = data.drop(['0'], axis=1)
    y = data[['0']].values.ravel()

    # Feature Scaling
    StdScaler = StandardScaler()
    X_scaled = StdScaler.fit_transform(X)

    # X_Train, x_test, Y_Train, y_test = train_test_split(X_scaled, y, test_size=0.25, random_state=0)

    pred = model.predict(X)
    print('Accuracy metrics are evaluated')

    # Accuracy
    accu_percent = accuracy_score(y, pred) * 100
    print("Accuracy obtained over the whole training set is %0.6f %% ." % (accu_percent))

    # Balanced Accuracy Score
    blnc = balanced_accuracy_score(y, pred) * 100
    print("balanced_accuracy_score: %0.6f %% ." % (blnc))


prediction()
