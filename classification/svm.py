import pandas as pd
import re
import sklearn
from sklearn import svm
import sklearn.cross_validation
from termcolor import colored
from classification import util

# refer to https://github.com/yassersouri/classify-text

METHOD = "SVM"


# METHOD = "KNN"
# METHOD = "MNB"


def main():
    origin_data = pd.read_csv('./assets/res_purchase_card_cleaned.csv', sep=",", error_bad_lines=False)
    predict_data = pd.read_csv('./assets/statementsLabeled.csv', sep=",", error_bad_lines=False)

    # print("Number of Columns:\n", origin_data.shape[1], "\n\n")
    # print("List of Columns:\n", ", ".join(origin_data.columns), "\n\n")
    # print("Data:\n", origin_data.head(), "\n\n")
    print("Size of train data(m):\n", origin_data.shape[0])

    origin_items = pd.DataFrame(origin_data, columns=['Description', 'Vendor', 'category_draft_1'])
    origin_text = origin_items["Description"] + " " + origin_items["Vendor"]

    test_items = pd.DataFrame(predict_data, columns=['TITLE', 'Category Draft-1'])
    test_text = [s[:-10] for s in test_items['TITLE']]

    all_title = origin_text.tolist() + test_text
    all_title = [re.sub(r'([^a-zA-Z0-9])+', ' ', s) for s in all_title]
    all_title = [re.sub(r'(\s)+', ' ', s) for s in all_title]
    word_counts = util.bagOfWords(all_title)

    all_label = origin_items["category_draft_1"].append(test_items["Category Draft-1"])
    all_label_code = all_label.astype('category').cat.codes
    all_label_code = all_label_code.tolist()
    origin_label = all_label_code[0:len(origin_text)]
    test_label = all_label_code[len(origin_text):]
    label_names = dict(enumerate(all_label.astype('category').astype('category').cat.categories)).values()

    # TFIDF
    tf_transformer = sklearn.feature_extraction.text.TfidfTransformer(use_idf=True).fit(word_counts)
    X = tf_transformer.transform(word_counts[0:len(origin_text)])
    X_predict = tf_transformer.transform(word_counts[len(origin_text):])

    # create classifier
    if METHOD == "MNB":
        clf = sklearn.naive_bayes.MultinomialNB()
    elif METHOD == "SVM":
        clf = sklearn.svm.LinearSVC()
    else:
        n_neighbors = 11
        weights = 'uniform'
        # weights = 'distance'
        clf = sklearn.neighbors.KNeighborsClassifier(n_neighbors, weights=weights)

    # test the classifier
    print(colored('Testing classifier with train-test split', 'magenta', attrs=['bold']))
    validation_classifier(X, origin_label, clf, test_size=0.2, y_names=label_names, confusion=False)
    test_classifier(X, origin_label, X_predict, test_label, clf, y_names=label_names, confusion=False)


def validation_classifier(X, y, clf, test_size=0.4, y_names=None, confusion=False):
    # train-test split
    print('test size is: %2.0f%%' % (test_size * 100))
    X_train, X_test, y_train, y_test = sklearn.cross_validation.train_test_split(X, y, test_size=test_size)

    clf.fit(X_train, y_train)
    y_predicted = clf.predict(X_test)

    if not confusion:
        print(colored('Classification report:', 'magenta', attrs=['bold']))
        print(sklearn.metrics.classification_report(y_test, y_predicted, target_names=y_names))
    else:
        print(colored('Confusion Matrix:', 'magenta', attrs=['bold']))
        print(sklearn.metrics.confusion_matrix(y_test, y_predicted))


def test_classifier(X, y, X_test, y_test, clf, y_names=None, confusion=False):
    clf.fit(X, y)
    y_predicted = clf.predict(X_test)

    if not confusion:
        print(colored('Classification report:', 'magenta', attrs=['bold']))
        print(sklearn.metrics.classification_report(y_test, y_predicted, target_names=y_names))
    else:
        print(colored('Confusion Matrix:', 'magenta', attrs=['bold']))
        print(sklearn.metrics.confusion_matrix(y_test, y_predicted))


if __name__ == '__main__':
    main()

# KNN & MNB
# Memory Error

# SVM
# Validation report:
#                         precision    recall  f1-score   support
#
#            Advertising       0.96      0.89      0.92       339
#             Air Travel       1.00      1.00      1.00      5909
#         Alcohol & Bars       1.00      1.00      1.00         2
#              Amusement       0.94      1.00      0.97        17
#                   Arts       0.91      0.82      0.86       119
#           Auto Payment       0.98      0.94      0.96       128
#                Bicycle       1.00      1.00      1.00        23
#       Books & Supplies       0.93      0.97      0.95      6479
#      Business Services       0.91      0.90      0.91      2669
#                Charity       0.91      0.88      0.89       771
#               Clothing       0.92      0.92      0.92       538
#           Coffee Shops       1.00      1.00      1.00         5
#                 Doctor       0.91      0.87      0.89       281
#              Education       0.94      0.89      0.92       376
# Electronics & Software       0.85      0.90      0.88      5076
#          Entertainment       0.78      0.88      0.82        32
#                Eyecare       1.00      0.82      0.90        11
#              Fast Food       0.96      0.96      0.96       878
#         Fees & Charges       1.00      1.00      1.00       250
#          Food & Dining       0.91      0.95      0.93        81
#            Furnishings       0.86      0.90      0.88       456
#             Gas & Fuel       0.99      0.96      0.98       387
#                   Gift       0.92      0.83      0.87       196
#              Groceries       0.99      0.99      0.99      3620
#                    Gym       0.67      0.40      0.50         5
#                   Hair       0.96      0.95      0.95       487
#                Hobbies       0.96      0.96      0.96      3714
#       Home Improvement       0.97      0.97      0.97      2302
#          Home Services       1.00      0.99      0.99      2655
#          Home Supplies       1.00      0.99      1.00      4269
#                  Hotel       1.00      0.99      1.00       142
#               Internet       0.96      1.00      0.98       128
#                Laundry       0.96      0.94      0.95       917
#          Lawn & Garden       1.00      0.44      0.62         9
#                  Legal       0.94      0.92      0.93      3018
#         Life Insurance       0.96      1.00      0.98      2018
#          Misc Expenses       0.94      0.77      0.85        22
#           Mobile Phone       1.00      0.82      0.90        17
#             Motorcycle       0.97      0.96      0.96       245
#          Movies & DVDs       0.95      0.94      0.95       667
#                  Music       0.98      0.97      0.98      6457
# Newspapers & Magazines       0.94      0.94      0.94        49
#        Office Supplies       0.77      0.83      0.80        12
#                Parking       0.95      0.94      0.95       271
#          Personal Care       0.96      0.99      0.98       134
#    Pet Food & Supplies       0.96      0.90      0.93       699
#               Pharmacy       0.99      0.95      0.97       253
#               Printing       0.94      0.91      0.92       846
#  Public Transportation       0.98      0.97      0.97      1996
#      Rental Car & Taxi       0.99      0.99      0.99      2180
#            Restaurants       1.00      1.00      1.00      2087
#        Service & Parts       0.97      0.94      0.96      3306
#            Service Fee       1.00      1.00      1.00         1
#               Shipping       0.90      0.88      0.89       396
#               Shopping       1.00      0.95      0.97        55
#          Spa & Massage       0.99      0.88      0.93       726
#         Sporting Goods       0.98      0.99      0.99       965
#                 Sports       0.82      0.77      0.79        30
#                Storage       0.94      0.89      0.91        70
#
#            avg / total       0.96      0.96      0.96     69791


# test report
#                         precision    recall  f1-score   support
#
#            Advertising       0.50      0.50      0.50         2
#             Air Travel       0.00      0.00      0.00         2
#         Alcohol & Bars       0.00      0.00      0.00         2
#              Amusement       0.00      0.00      0.00         1
#                   Arts       0.00      0.00      0.00         1
#           Auto Payment       0.00      0.00      0.00         0
#                Bicycle       0.67      0.40      0.50         5
#       Books & Supplies       0.00      0.00      0.00        27
#      Business Services       0.00      0.00      0.00         5
#                Charity       0.00      0.00      0.00         3
#               Clothing       0.00      0.00      0.00         3
#           Coffee Shops       0.12      0.20      0.15         5
#                 Doctor       0.00      0.00      0.00        11
#              Education       0.00      0.00      0.00         0
# Electronics & Software       0.00      0.00      0.00         0
#          Entertainment       0.00      0.00      0.00        59
#                Eyecare       0.00      0.00      0.00         1
#              Fast Food       0.00      0.00      0.00         0
#         Fees & Charges       0.00      0.00      0.00         2
#          Food & Dining       0.00      0.00      0.00         0
#            Furnishings       0.00      0.00      0.00         2
#             Gas & Fuel       1.00      1.00      1.00         2
#                   Gift       0.00      0.00      0.00         1
#              Groceries       0.00      0.00      0.00         1
#                    Gym       0.00      0.00      0.00         0
#                   Hair       0.00      0.00      0.00         1
#                Hobbies       0.00      0.00      0.00         1
#       Home Improvement       0.67      0.12      0.20        17
#          Home Services       0.00      0.00      0.00         0
#          Home Supplies       0.00      0.00      0.00         0
#                  Hotel       0.00      0.00      0.00         3
#               Internet       0.00      0.00      0.00         3
#                Laundry       0.00      0.00      0.00         0
#          Lawn & Garden       0.00      0.00      0.00         4
#                  Legal       1.00      0.17      0.29         6
#         Life Insurance       0.20      0.02      0.03       107
#          Misc Expenses       0.00      0.00      0.00         5
#           Mobile Phone       0.00      0.00      0.00         0
#             Motorcycle       0.11      0.03      0.05        33
#          Movies & DVDs       0.00      0.00      0.00         0
#                  Music       0.00      0.00      0.00         1
# Newspapers & Magazines       0.00      0.00      0.00         2
#        Office Supplies       0.00      0.00      0.00         1
#                Parking       0.00      0.00      0.00         2
#          Personal Care       0.00      0.00      0.00         5
#    Pet Food & Supplies       0.00      0.00      0.00         5
#
#            avg / total       0.15      0.04      0.05       331
