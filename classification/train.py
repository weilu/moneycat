import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import KFold, train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn import preprocessing, metrics
from sklearn.linear_model import SGDClassifier
from sklearn.externals import joblib
from sklearn.exceptions import UndefinedMetricWarning
from glob import glob
import boto3
import os
import pickle
from statistics import mean
import numpy as np

import warnings
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

s3 = boto3.client('s3')
N_FOLD = 10

LOSS_FNS = ['hinge', 'log'] # hinge = linear SVM, log = logistic regression
PENALTIES = ['l1', 'l2', 'elasticnet']

sgd_classifiers = []
for loss in LOSS_FNS:
    for penalty in PENALTIES:
        sgd_classifiers.append(SGDClassifier(loss=loss, penalty=penalty,
            random_state=123, max_iter=1000, tol=1e-3))
MNB_CLASSIFIERS = [MultinomialNB(alpha=.01), MultinomialNB(alpha=.1), MultinomialNB(alpha=.5), MultinomialNB(alpha=1)]
KNN_CLASSIFIERS = [KNeighborsClassifier(algorithm='auto')] # auto will try ‘ball_tree’, ‘kd_tree’, ‘brute’ and select the best
CLASSIFIERS = MNB_CLASSIFIERS + KNN_CLASSIFIERS + sgd_classifiers + [DecisionTreeClassifier(), RandomForestClassifier()]

def time_me(fn):
  def _wrapper(*args, **kwargs):
    start = time.clock()
    result = fn(*args, **kwargs)
    print ("%s cost %s second\n"%(fn.__name__, (time.clock() - start) / N_FOLD))
    return result
  return _wrapper


def read_data():
    return pd.read_csv('../cleaning/pdf-mining/out_wei_labelled_full.csv', sep=",")


@time_me
def cross_validate(X, y, classifier):
    kf = KFold(n_splits=N_FOLD, shuffle=True, random_state=42)
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
    print('%s produces an accuracy of %0.3f, and f1 score of %0.3f, train sample size: %d, feature size: %d'\
            % (classifier, accuracy, mean(f1s), len(X[train_index]), len(transformer.idf_)))
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


def export_model(classifier, transformer, label_trasformer, test_samples,
                 meta_data, report):
    joblib.dump(classifier, 'svm_classifier.pkl')
    joblib.dump(transformer, 'tfidf_transformer.pkl')
    joblib.dump(label_trasformer, 'label_transformer.pkl')
    test_samples.to_pickle('test_samples.pkl')
    pickle.dump(meta_data, open("meta.pkl", "wb"))

    # upload to s3
    model_bucket = 'cs4225-models'
    s3.put_object(Bucket=model_bucket, Body=report, Key='report.txt')
    for filename in glob('*.pkl'):
        s3.upload_file(filename, model_bucket, filename)
        os.remove(filename)
    print('\n\n============ Updated models on S3 ============')
    for obj in s3.list_objects(Bucket=model_bucket)['Contents']:
        print(obj['Key'], obj['LastModified'])


# Return model with the highest accuracy
def test_models(X, y):
    classifier_n_accuracies = map(lambda c: cross_validate(X, y, c), CLASSIFIERS)
    return sorted(classifier_n_accuracies, key=lambda pair: pair[1], reverse=True)[0]


def get_label_encoder(y, y_additional=pd.DataFrame()):
    le = preprocessing.LabelEncoder()
    y_raw = y.append(y_additional, ignore_index=True) if not y_additional.empty else y
    le.fit(y_raw)
    return le


def test_additional_train_data():
    gov_data = pd.read_csv('./assets/res_purchase_card_cleaned.csv', sep=",")
    print("gov data size:", gov_data.shape)

    data_dict = {'description': gov_data["Description"] + " " + gov_data["Vendor"],
                 'category': gov_data['category_draft_1']}
    gov_data = pd.DataFrame(data=data_dict)
    extra_X = gov_data['description']
    extra_y_raw = gov_data['category']

    personal_data = read_data()
    X = personal_data['description']
    y_raw = personal_data['category']

    le = get_label_encoder(y_raw, extra_y_raw)
    y = le.transform(y_raw)
    extra_y = le.transform(extra_y_raw)

    # leave out a small test subset for benchmarking
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=50,
                                                        random_state=123)
    X_train = X_train.append(extra_X, ignore_index=True)
    y_train = np.concatenate((y_train, extra_y))

    for classifier in CLASSIFIERS:
        transformer = train(X_train, y_train, classifier)
        pred = classifier.predict(transformer.transform(X_test))
        accuracy = metrics.accuracy_score(y_test, pred)
        f1 = metrics.f1_score(y_test, pred, average='weighted')
        meta_data = {'train_size': X_train.shape[0], 'accuracy': accuracy, 'f1': f1}
        print('%s produces an accuracy of %0.3f, and f1 score of %0.3f'\
                % (classifier, accuracy, f1))

def train_pure_personal_data():
    personal_data = read_data()
    print("data size:", personal_data.shape)

    y_raw = personal_data['category']
    le = get_label_encoder(y_raw)
    y = le.transform(y_raw)
    X = personal_data['description']

    # report cross validation accuracy
    classifier, accuracy = test_models(X, y)
    print('Using {} with accuracy {}'.format(classifier, accuracy))

    # leave out a small test subset for benchmarking
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=50,
                                                        random_state=123)
    transformer = train(X_train, y_train, classifier)
    pred = classifier.predict(transformer.transform(X_test))
    report = metrics.classification_report(y_test, pred, target_names=list(le.classes_))
    # print(report)

    accuracy = metrics.accuracy_score(y_test, pred)
    f1 = metrics.f1_score(y_test, pred, average='weighted')
    test_samples = pd.DataFrame(data={'X': X_test, 'y': y_test})
    meta_data = {'train_size': X_train.shape[0], 'accuracy': accuracy, 'f1': f1}
    print("test sample size: %d, accuracy: %0.3f, f1 score: %0.3f" \
            % (X_test.shape[0], accuracy, f1))
    # export_model(classifier, transformer, le, test_samples, meta_data, report)


if __name__ == '__main__':
    # test_additional_train_data()
    train_pure_personal_data()
