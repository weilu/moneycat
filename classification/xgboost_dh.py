import pandas as pd
import numpy as np
import time
from statistics import mean
from sklearn import preprocessing, metrics
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold
import xgboost as xgb

N_FOLD = 10

def time_me(fn):
  def _wrapper(*args, **kwargs):
    start = time.clock()
    fn(*args, **kwargs)
    print ("\n%s cost %s second"%(fn.__name__, (time.clock() - start) / N_FOLD))
  return _wrapper


personal_data = pd.read_csv('assets/out_wei_labelled_full.csv', sep=",")
categories = category = pd.read_csv('assets/category.csv').category
lbl = LabelEncoder()
lbl.fit(category)

y = lbl.transform(personal_data['category'])
X = personal_data['description']

params = {'objective': 'multi:softmax', 'num_class': len(category)}
params['silent'] = 1
num_round = 30

@time_me
def cross_validate(X, y):
    kf = KFold(n_splits=N_FOLD, shuffle=True, random_state=42)
    accuracies = []
    f1s = []
    for train_index, test_index in kf.split(X):
        transformer = TfidfVectorizer(max_df=1.0)
        X_train = transformer.fit_transform(X[train_index])
        X_test = transformer.transform(X[test_index])

        xg_train = xgb.DMatrix(X_train, label=y[train_index])
        xg_test = xgb.DMatrix(X_test, label=y[test_index])

        model = xgb.train(params, xg_train, num_round)
        pred = model.predict(xg_test)

        accuracy = metrics.accuracy_score(y[test_index], pred)
        f1 = metrics.f1_score(y[test_index], pred, average='weighted')
        # print("test sample size: %d, accuracy: %0.3f, f1 score: %0.3f" \
        #         % (X_test.shape[0], accuracy, f1))
        print('original model accuracy of %0.3f, and f1 score of %0.3f' \
              % (accuracy, f1))

        accuracies.append(accuracy)
        f1s.append(f1)
    accuracy = mean(accuracies)
    print('%s produces an accuracy of %0.3f, and f1 score of %0.3f'\
            % (model, accuracy, mean(f1s)))

if __name__ == '__main__':
    cross_validate(X, y)