import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import KFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn import metrics
from termcolor import colored


def read_data():
    return pd.read_csv('./assets/statementsLabeled.csv', sep=",")


def cross_validate(X, y):
    fold = 4
    sum = 0
    kf = KFold(n_splits=fold, shuffle=True, random_state=42)
    classifier, transformer = None, None
    for train_index, test_index in kf.split(X):
        classifier, transformer = train_tfidf(X[train_index], y[train_index])
        X_test = transformer.transform(X[test_index])
        pred = classifier.predict(X_test)
        score = metrics.accuracy_score(y[test_index], pred)
        print("test sample size: %d, accuracy: %0.3f" % (X_test.shape[0], score))
        sum += score
    average = sum/fold
    print(f'Naive bayes with tfidf produces an accuracy of {average}')
    return (classifier, transformer)


def train_tfidf(X, y):
    max_df = 1.0
    transformer = TfidfVectorizer(max_df=max_df)
    X_train = transformer.fit_transform(X)
    classifier = MultinomialNB(alpha=.01)
    # classifier = BernoulliNB(alpha=.01)
    classifier.fit(X_train, y)

    print("train n_samples: %d, n_features: %d" % X_train.shape)
    if len(transformer.stop_words_):
        print("idf stop words: ")
        print(" ".join(transformer.stop_words_))

    return (classifier, transformer)


if __name__ == '__main__':
    personal_data = read_data()
    print("data size:", personal_data.shape)

    le = preprocessing.LabelEncoder()
    y = le.fit_transform(personal_data['Category Draft-1'])
    X = personal_data['TITLE']
    # report cross validation accuracy
    classifier, transformer = cross_validate(X, y)

    # per category support is way to small in test set,
    # so use the entire data set to produce classification report
    pred = classifier.predict(transformer.transform(X))
    print(colored('Classification report:', 'magenta', attrs=['bold']))
    print(metrics.classification_report(y, pred, target_names=list(le.classes_)))

# data size: (331, 4)
# train n_samples: 248, n_features: 355
# test sample size: 83, accuracy: 0.771
# train n_samples: 248, n_features: 357
# test sample size: 83, accuracy: 0.807
# train n_samples: 248, n_features: 329
# test sample size: 83, accuracy: 0.651
# train n_samples: 249, n_features: 340
# test sample size: 82, accuracy: 0.695
# Naive bayes with tfidf produces an accuracy of 0.7310094034675287
# Classification report:
#                         precision    recall  f1-score   support
#
#             Air Travel       1.00      0.50      0.67         2
#         Alcohol & Bars       1.00      1.00      1.00         2
#              Amusement       1.00      1.00      1.00         2
#                   Arts       0.00      0.00      0.00         1
#       Books & Supplies       1.00      1.00      1.00         1
#               Clothing       0.80      0.80      0.80         5
#           Coffee Shops       0.92      0.89      0.91        27
#              Education       0.83      1.00      0.91         5
# Electronics & Software       1.00      1.00      1.00         3
#          Entertainment       1.00      1.00      1.00         3
#              Fast Food       1.00      0.80      0.89         5
#          Food & Dining       1.00      0.82      0.90        11
#              Groceries       0.93      0.93      0.93        59
#                    Gym       0.00      0.00      0.00         1
#       Home Improvement       1.00      1.00      1.00         2
#          Home Supplies       0.67      1.00      0.80         2
#                  Hotel       1.00      1.00      1.00         2
#               Internet       0.50      1.00      0.67         1
#         Life Insurance       1.00      1.00      1.00         1
#           Mobile Phone       1.00      1.00      1.00         1
#          Movies & DVDs       1.00      1.00      1.00         1
#                  Music       1.00      0.88      0.94        17
#          Personal Care       1.00      1.00      1.00         3
#    Pet Food & Supplies       1.00      1.00      1.00         3
#  Public Transportation       1.00      0.75      0.86         4
#      Rental Car & Taxi       1.00      1.00      1.00         6
#            Restaurants       0.94      0.96      0.95       107
#            Service Fee       1.00      0.80      0.89         5
#               Shopping       0.91      0.97      0.94        33
#                 Sports       1.00      1.00      1.00         1
#                 Travel       1.00      1.00      1.00         2
#                Tuition       0.00      0.00      0.00         1
#          Uncategorized       1.00      1.00      1.00         2
#              Utilities       0.60      0.60      0.60         5
#               Vacation       0.56      1.00      0.71         5
#
#            avg / total       0.92      0.92      0.92       331
