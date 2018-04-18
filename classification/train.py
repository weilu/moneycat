import pandas as pd
from sklearn.model_selection import KFold, train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn import preprocessing, metrics, svm
from sklearn.linear_model import SGDClassifier
from sklearn.externals import joblib
from glob import glob
import boto3
import os

s3 = boto3.client('s3')


def read_data():
    return pd.read_csv('../cleaning/pdf-mining/out_wei_labelled_full.csv', sep=",")


def cross_validate(X, y):
    fold = 10
    sum = 0
    kf = KFold(n_splits=fold, shuffle=True, random_state=42)
    classifier, transformer = None, None
    for train_index, test_index in kf.split(X):
        classifier, transformer = train(X[train_index], y[train_index])
        X_test = transformer.transform(X[test_index])
        pred = classifier.predict(X_test)
        score = metrics.accuracy_score(y[test_index], pred)
        print("test sample size: %d, accuracy: %0.3f" % (X_test.shape[0], score))
        sum += score
    average = sum/fold
    print(f'{classifier} produces an accuracy of {average}')
    return (classifier, transformer)


def train(X, y):
    max_df = 1.0
    transformer = TfidfVectorizer(max_df=max_df)
    X_train = transformer.fit_transform(X)
    # classifier = MultinomialNB(alpha=.01)
    # classifier = BernoulliNB(alpha=.01)
    # classifier = svm.LinearSVC()
    classifier = SGDClassifier(random_state=123, max_iter=1000, tol=1e-3)
    classifier.fit(X_train, y)

    print("train n_samples: %d, n_features: %d" % X_train.shape)
    if len(transformer.stop_words_):
        print("idf stop words: ")
        print(" ".join(transformer.stop_words_))

    return (classifier, transformer)


def export_model(classifier, transformer, label_trasformer, test_samples, report):
    joblib.dump(classifier, 'svm_classifier.pkl')
    joblib.dump(transformer, 'tfidf_transformer.pkl')
    joblib.dump(label_trasformer, 'label_transformer.pkl')
    test_samples.to_pickle('test_samples.pkl')

    # upload to s3
    model_bucket = 'cs4225-models'
    s3.put_object(Bucket=model_bucket, Body=report, Key='report.txt')
    for filename in glob('*.pkl'):
        s3.upload_file(filename, model_bucket, filename)
        os.remove(filename)
    print('\n\n============ Updated models on S3 ============')
    for obj in s3.list_objects(Bucket=model_bucket)['Contents']:
        print(obj['Key'], obj['LastModified'])

if __name__ == '__main__':
    personal_data = read_data()
    print("data size:", personal_data.shape)

    le = preprocessing.LabelEncoder()
    y = le.fit_transform(personal_data['category'])
    X = personal_data['description']
    # report cross validation accuracy
    cross_validate(X, y)

    # leave out a small test subset for benchmarking
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=50,
                                                        random_state=42)
    classifier, transformer = train(X_train, y_train)
    pred = classifier.predict(transformer.transform(X_test))
    report = metrics.classification_report(y_test, pred, target_names=list(le.classes_))
    print(report)

    test_samples = pd.DataFrame(data={'X': X_test, 'y': y_test})
    export_model(classifier, transformer, le, test_samples, report)
