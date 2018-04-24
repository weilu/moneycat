import pandas as pd
import numpy as np
from statistics import mean
from sklearn import metrics
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
import xgboost as xgb

personal_data = pd.read_csv('assets/out_wei_labelled_full.csv', sep=",")
categories = category = pd.read_csv('assets/category.csv').category
lbl = LabelEncoder()
lbl.fit(category)

y = lbl.transform(personal_data['category'])
X = personal_data['description']



CLASSIFIERS = [SGDClassifier(penalty='l2', loss='hinge'),
               SGDClassifier(penalty='l1', loss='log'),
               DecisionTreeClassifier(),
               RandomForestClassifier()
               ]

def cross_validate(X, y, classifier):
    fold = 10
    kf = KFold(n_splits=fold, shuffle=True, random_state=42)
    accuracies = []
    f1s = []
    for train_index, test_index in kf.split(X):
        transformer = train(X[train_index], y[train_index], classifier)
        X_test = transformer.transform(X[test_index])
        pred = classifier.predict(X_test)
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

def train(X, y, classifier):
    max_df = 1.0
    transformer = TfidfVectorizer(max_df=max_df)
    X_train = transformer.fit_transform(X)
    classifier.fit(X_train, y)

    # print("train n_samples: %d, n_features: %d" % X_train.shape)
    if len(transformer.stop_words_):
        print("idf stop words: ")
        print(" ".join(transformer.stop_words_))

    return transformer

def test_models(X, y):
    classifier_n_accuracies = map(lambda c: cross_validate(X, y, c), CLASSIFIERS)
    return sorted(classifier_n_accuracies, key=lambda pair: pair[1], reverse=True)[0]

if __name__ == '__main__':
    # report cross validation accuracy
    classifier, accuracy = test_models(X, y)
    print('Using {} with accuracy {}'.format(classifier, accuracy))