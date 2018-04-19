import pandas as pd
from sklearn.model_selection import KFold, train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn import preprocessing, metrics
from sklearn.linear_model import SGDClassifier
from sklearn.externals import joblib
from glob import glob
import boto3
import os
import pickle
from statistics import mean

s3 = boto3.client('s3')


def read_data():
    return pd.read_csv('../cleaning/pdf-mining/out_wei_labelled_full.csv', sep=",")


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
        print("test sample size: %d, accuracy: %0.3f, f1 score: %0.3f" \
                % (X_test.shape[0], accuracy, f1))
        accuracies.append(accuracy)
        f1s.append(f1)
    accuracy = mean(accuracies)
    print('%s produces an accuracy of %0.3f, and f1 score of %0.3f'\
            % (classifier.__class__.__name__, accuracy, mean(f1s)))
    return (classifier, accuracy)


def train(X, y, classifier):
    max_df = 1.0
    transformer = TfidfVectorizer(max_df=max_df)
    X_train = transformer.fit_transform(X)
    classifier.fit(X_train, y)

    print("train n_samples: %d, n_features: %d" % X_train.shape)
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
    classifiers = [MultinomialNB(alpha=.01),
                   KNeighborsClassifier(),
                   SGDClassifier(random_state=123, max_iter=1000, tol=1e-3)]
    classifier_n_accuracies = map(lambda c: cross_validate(X, y, c), classifiers)
    return sorted(classifier_n_accuracies, key=lambda pair: pair[1], reverse=True)[0]


if __name__ == '__main__':
    personal_data = read_data()
    print("data size:", personal_data.shape)

    le = preprocessing.LabelEncoder()
    y = le.fit_transform(personal_data['category'])
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
    print(report)

    score = metrics.accuracy_score(y_test, pred)
    test_samples = pd.DataFrame(data={'X': X_test, 'y': y_test})
    meta_data = {'train_size': X_train.shape[0], 'accuracy': score}
    # export_model(classifier, transformer, le, test_samples, meta_data, report)
