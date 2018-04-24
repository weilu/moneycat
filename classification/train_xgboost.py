import pandas as pd
import numpy as np
import time
import xgboost as xgb
from sklearn.model_selection import train_test_split as ttsplit
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn import metrics

def time_me(fn):
  def _wrapper(*args, **kwargs):
    start = time.clock()
    fn(*args, **kwargs)
    print ("\n%s cost %s second"%(fn.__name__, time.clock() - start))
  return _wrapper

personal_data = pd.read_csv('assets/out_wei_labelled_full.csv', sep=",")
categories = category = pd.read_csv('assets/category.csv').category
lbl = LabelEncoder()
lbl.fit(category)

y = lbl.transform(personal_data['category'])
X = personal_data['description']

transformer = TfidfVectorizer(max_df=1.0)
X = transformer.fit_transform(X).toarray()

@time_me
def train_incremental(X, y):
    # split data into training and testing sets
    # then split training set in half
    # the first part is used as orginal data
    # the second part is used as incremental data
    X_train, X_test, y_train, y_test = ttsplit(X, y, test_size=0.1, random_state=0)
    X_train_origin, X_train_incremental, y_train_origin, y_train_incremental = ttsplit(X_train,
                                                         y_train,
                                                         test_size=0.5,
                                                         random_state=0)

    xg_train_origin = xgb.DMatrix(X_train_origin, label=y_train_origin)
    xg_train_incremental = xgb.DMatrix(X_train_incremental, label=y_train_incremental)
    xg_test = xgb.DMatrix(X_test, label=y_test)

    # ================= xgboost classification model ====================#
    params = {'objective': 'multi:softmax', 'num_class': len(category)}
    params['silent'] = 1
    num_round = 30

    model_origin = xgb.train(params, xg_train_origin, num_round)
    model_origin.save_model('xgb_model.model')

    # ================= train two versions of the model =====================#
    model_none_incremental = xgb.train(params, xg_train_incremental, num_round)
    model_incremental = xgb.train(params, xg_train_incremental, num_round, xgb_model='xgb_model.model')

    # benchmark
    pred_origin = model_origin.predict(xg_test)
    score = metrics.accuracy_score(y_test, pred_origin)
    f1 = metrics.f1_score(y_test, pred_origin, average='weighted')
    print('original model accuracy of %0.3f, and f1 score of %0.3f' \
          % (score, f1))

    # "before"
    pred_none_incremental = model_none_incremental.predict(xg_test)
    score = metrics.accuracy_score(y_test, pred_none_incremental)
    f1 = metrics.f1_score(y_test, pred_none_incremental, average='weighted')
    print('none incremental model accuracy of %0.3f, and f1 score of %0.3f' \
          % (score, f1))

    # "after"
    pred_incremental = model_incremental.predict(xg_test)
    score = metrics.accuracy_score(y_test, pred_incremental)
    f1 = metrics.f1_score(y_test, pred_incremental, average='weighted')
    print('incremental model accuracy of %0.3f, and f1 score of %0.3f' \
          % (score, f1))

if __name__ == '__main__':
    # xgboost model costs quite a long time for training whereas the accuracy of it is unexpected low.
    train_incremental(X, y)