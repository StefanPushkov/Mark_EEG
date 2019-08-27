from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np
import config as cf
from sklearn.model_selection import train_test_split

data = pd.read_csv("../" + cf.prepared_data)
X = data.drop(['0'], axis=1)
y = data[['0']].values.ravel()

# Splitting the dataset into the Training set and Test set
X_Train, x_test, Y_Train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# Feature Scaling
StdScaler = StandardScaler()
X_Train = StdScaler.fit_transform(X_Train)

# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]
# Method of selecting samples for training each tree
bootstrap = [True, False]

random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}

clf = RandomForestClassifier(random_state=0)

rf_random = RandomizedSearchCV(estimator = clf, param_distributions=random_grid,
                               n_iter=20, cv=3, verbose=1, random_state=42, n_jobs=-1)

# Fit the random search model
rf_random.fit(X_Train, Y_Train)
best_p = rf_random.best_params_
best_r = rf_random.best_score_


import json
with open("../CV_result/cv_randomForest.txt", "w") as f:
    f.write('Best Params: \n')
    f.write(json.dumps(best_p))
    f.write('\nBest Accuracy: \n')
    f.write(json.dumps(best_r))
    f.close()