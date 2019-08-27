from joblib import dump, load
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
X_Train, x_test, Y_Train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

def prediction(data: str):
    model = load('../models/RandomForest_model.joblib')
    data = pd.read_csv(data)
    data = data.loc[:]
    X = data.drop(['0'], axis=1)
    y = data[['0']].values.ravel()

