import pandas as pd
import numpy as np
import time
from statistics import mean
from sklearn import preprocessing, metrics
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
import xgboost as xgb

N_FOLD = 10

def time_me(fn):
  def _wrapper(*args, **kwargs):
    start = time.clock()
    fn(*args, **kwargs)
    print ("\n%s cost %s second"%(fn.__name__, (time.clock() - start) / N_FOLD))
  return _wrapper

@time_me
def cross_validate(X, y, classifier):
    fold = N_FOLD
    kf = KFold(n_splits=fold, shuffle=True, random_state=42)
    accuracies = []
    f1s = []
    for train_index, test_index in kf.split(X):
        transformer = train(X, y, classifier, train_index)
        pred = predict(X, classifier, transformer, test_index)
        accuracy = metrics.accuracy_score(y[test_index], pred)
        f1 = metrics.f1_score(y[test_index], pred, average='weighted')
        # print("test sample size: %d, accuracy: %0.3f, f1 score: %0.3f" \
        #         % (X_test.shape[0], accuracy, f1))
        accuracies.append(accuracy)
        f1s.append(f1)
    accuracy = mean(accuracies)
    print('%s produces an accuracy of %0.3f, and f1 score of %0.3f'\
            % (classifier, accuracy, mean(f1s)))
    return (classifier, accuracy)

def train(X, y, classifier, train_index):
    max_df = 1.0
    transformer = TfidfVectorizer(max_df=max_df)
    tfidf_X = transformer.fit_transform(X[train_index].values)
    tfidf_X = pd.DataFrame(tfidf_X.toarray())

    X_train = tfidf_X
    if len(additional_feature_names) > 0:
        additionals = processed_features.iloc[train_index][additional_feature_names]
        additionals.index = range(len(additionals.index))
        X_train = pd.concat((tfidf_X, additionals), axis=1).reset_index(drop=True)
    classifier.fit(X_train.values, y[train_index])

    # print("train n_samples: %d, n_features: %d" % X_train.shape)
    if len(transformer.stop_words_):
        print("idf stop words: ")
        print(" ".join(transformer.stop_words_))

    return transformer

def predict(X, classifier, transformer, test_index):
    tfidf_X = transformer.transform(X[test_index].values)
    tfidf_X = pd.DataFrame(tfidf_X.toarray())

    X_test = tfidf_X
    if len(additional_feature_names) > 0:
        additionals = processed_features.iloc[test_index][additional_feature_names]
        additionals.index = range(len(additionals.index))
        X_test = pd.concat((tfidf_X, additionals), axis=1).reset_index(drop=True)

    return classifier.predict(X_test.values)

def process_features():
    amount = personal_data['amount'].str.replace(',', '')
    amount = amount.str.split(' ', expand=True)

    amount[0][amount[1].notnull()] = amount[0][amount[1].notnull()].apply(lambda x: float(x) * float(-1.0))
    amount[0] = amount[0].apply(float)

    foreign_amount = personal_data['foreign_amount']
    foreign_amount = foreign_amount.str.replace(',', '')
    foreign_amount = foreign_amount.str.split(' ', expand=True)
    foreign_amount[1] = foreign_amount[1].apply(float)

    processed = pd.DataFrame()
    processed['amount'] = amount[0]
    processed['amount_type'] = amount[1]
    processed['foreign_amount'] = foreign_amount[1]
    processed['foreign_amount_currency'] = foreign_amount[0]

    processed["amount_type"] = processed["amount_type"].fillna("None")
    processed["foreign_amount_currency"] = processed["foreign_amount_currency"].fillna("None")
    processed["foreign_amount"] = processed["foreign_amount"].fillna(0)

    print('The list of amount_type is:\t', processed["amount_type"].unique())
    print('The list of foreign_amount_currency is:\t', processed["foreign_amount_currency"].unique())

    # label encode the object features
    cols = processed.select_dtypes(include=['object']).columns.values
    for c in cols:
        lbl = LabelEncoder()
        lbl.fit(list(processed[c].values))
        processed[c] = lbl.transform(list(processed[c].values))

    # scaling the numerical features
    cols = processed.select_dtypes(include=['int64', 'float64']).columns.values
    for c in cols:
        processed[c] = preprocessing.normalize(processed.values, norm='l2')

    return processed

def test_models(X, y):
    classifier_n_accuracies = map(lambda c: cross_validate(X, y, c), CLASSIFIERS)
    return sorted(classifier_n_accuracies, key=lambda pair: pair[1], reverse=True)[0]


personal_data = pd.read_csv('assets/out_wei_labelled_full.csv', sep=",")
categories = category = pd.read_csv('assets/category.csv').category
lbl = LabelEncoder()
lbl.fit(category)

y = lbl.transform(personal_data['category'])
X = personal_data['description']

processed_features = process_features()
additional_feature_names = ['amount', 'amount_type']


CLASSIFIERS = [SGDClassifier(penalty='l2', loss='hinge'),
               SGDClassifier(penalty='l1', loss='log'),
               SGDClassifier(penalty='elasticnet', loss='log'),
               DecisionTreeClassifier(),
               RandomForestClassifier()
               ]

if __name__ == '__main__':
    # report cross validation accuracy
    classifier, accuracy = test_models(X, y)
    print('Using {} with accuracy {} with additional features {}'.format(classifier, accuracy, additional_feature_names))