import re

import pandas as pd
import sklearn
import sklearn.cross_validation
import util
from sklearn import svm
from sklearn.model_selection import train_test_split
from termcolor import colored
import warnings

warnings.filterwarnings('ignore')
# refer to https://github.com/yassersouri/classify-text

METHOD = "SVM"


# METHOD = "KNN"
# METHOD = "MNB"


def main():
    origin_data = pd.read_csv('./assets/res_purchase_card_cleaned.csv', sep=",", error_bad_lines=False)
    predict_data = pd.read_csv('./assets/out_wei_labelled_full.csv', sep=",", error_bad_lines=False)

    # print("Number of Columns:\n", origin_data.shape[1], "\n\n")
    # print("List of Columns:\n", ", ".join(origin_data.columns), "\n\n")
    # print("Data:\n", origin_data.head(), "\n\n")
    # print("Size of train data(m):\n", origin_data.shape[0])

    origin_items = pd.DataFrame(origin_data, columns=['Description', 'Vendor', 'category_draft_1'])
    origin_items = origin_items.sample(frac=0.05, replace=True, random_state=4252)
    origin_text = origin_items["Description"] + " " + origin_items["Vendor"]
    print("Size of train data(m):\n", origin_text.shape[0])

    test_items = pd.DataFrame(predict_data, columns=['description', 'category'])

    a_test_items, b_test_items = train_test_split(test_items, test_size=0.2)
    a_test_text = a_test_items["description"]
    b_test_text = b_test_items["description"]

    all_title = origin_text.tolist() + a_test_text.tolist() + b_test_text.tolist()
    all_title = [re.sub(r'([^a-zA-Z0-9])+', ' ', s) for s in all_title]
    all_title = [re.sub(r'(\s)+', ' ', s) for s in all_title]
    word_counts = util.bagOfWords(all_title)

    all_label = origin_items["category_draft_1"].append(a_test_items["category"]).append(b_test_items["category"])
    all_label_code = all_label.astype('category').cat.codes
    all_label_code = all_label_code.tolist()
    origin_label = all_label_code[0:len(origin_text) + len(a_test_text)]
    test_label = all_label_code[len(origin_text) + len(a_test_text):]
    label_names = dict(enumerate(all_label.astype('category').astype('category').cat.categories)).values()

    # TFIDF
    tf_transformer = sklearn.feature_extraction.text.TfidfTransformer(use_idf=True).fit(word_counts)
    X = tf_transformer.transform(word_counts[0:len(origin_text) + len(a_test_text)])
    X_predict = tf_transformer.transform(word_counts[len(origin_text) + len(a_test_text):])

    # create classifier
    if METHOD == "MNB":
        clf = sklearn.naive_bayes.MultinomialNB()
    elif METHOD == "SVM":
        clf = svm.LinearSVC()
    else:
        n_neighbors = 11
        weights = 'uniform'
        # weights = 'distance'
        clf = sklearn.neighbors.KNeighborsClassifier(n_neighbors, weights=weights)

    # test the classifier
    print(colored('Testing classifier with train-test split', 'magenta', attrs=['bold']))
    validation_classifier(X, origin_label, clf, test_size=0.1, y_names=label_names, confusion=False)
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

#
# Size of train data(m):
#  996
# Testing classifier with train-test split
# test size is: 10%
# Classification report:
#                         precision    recall  f1-score   support
#
#             Air Travel       1.00      0.50      0.67         2 ++
#         Alcohol & Bars       0.00      0.00      0.00         0 ++
#              Amusement       0.00      0.00      0.00         2 ++
#                   Arts       1.00      1.00      1.00         1 ++
#               Bank Fee       1.00      0.89      0.94        19
#                  Bonus       1.00      1.00      1.00         1 ++
#               Clothing       0.91      0.67      0.77        15
#           Coffee Shops       1.00      1.00      1.00         1 ++
#    Credit Card Payment       1.00      1.00      1.00         2 ++
#               Delivery       0.50      1.00      0.67         1 ++
#                Dentist       0.00      0.00      0.00         2 ++
#                 Doctor       1.00      0.80      0.89         5 ++
#              Education       1.00      1.00      1.00         1 ++
# Electronics & Software       1.00      1.00      1.00         1 ++
#          Entertainment       0.00      0.00      0.00         2 ++
#              Fast Food       1.00      1.00      1.00         1 ++
#          Food & Dining       1.00      1.00      1.00         1 ++
#            Furnishings       0.00      0.00      0.00         1 ++
#             Gas & Fuel       1.00      1.00      1.00         3 ++
#                   Gift       1.00      1.00      1.00         1 ++
#              Groceries       0.00      0.00      0.00         1 ++
#                    Gym       0.77      1.00      0.87        17
#       Home Improvement       0.43      0.90      0.58        10
#          Home Services       1.00      1.00      1.00         4 ++
#          Home Supplies       0.00      0.00      0.00         1 ++
#                  Hotel       1.00      1.00      1.00         2 ++
#                 Income       0.00      0.00      0.00         2 ++
#        Interest Income       1.00      1.00      1.00         1 ++
#
#            avg / total       0.78      0.79      0.77       100

#
#
# Classification report:
#                         precision    recall  f1-score   support
#
#            Advertising       0.94      0.91      0.93       292
#             Air Travel       1.00      1.00      1.00      5905
#         Alcohol & Bars       0.00      0.00      0.00         1
#              Amusement       1.00      0.89      0.94        18
#                   Arts       0.91      0.89      0.90       149
#           Auto Payment       0.98      0.93      0.95       123
#               Bank Fee       1.00      0.96      0.98        23
#                Bicycle       0.93      0.97      0.95      6327
#                  Bonus       0.88      0.91      0.89      2693
#       Books & Supplies       0.92      0.89      0.91       806
#      Business Services       0.94      0.93      0.94       574
#                Charity       1.00      1.00      1.00         3
#               Clothing       0.93      0.87      0.90       289
#           Coffee Shops       0.91      0.91      0.91       397
#    Credit Card Payment       0.88      0.89      0.89      5166
#               Delivery       0.88      0.78      0.83        37
#                Dentist       1.00      0.75      0.86         8
#                 Doctor       0.95      0.94      0.95       854
#              Education       1.00      0.99      1.00       252
# Electronics & Software       0.99      0.90      0.94        96
#          Entertainment       0.88      0.88      0.88       415
#                Eyecare       0.98      0.97      0.98       395
#              Fast Food       0.96      0.84      0.89       231
#         Fees & Charges       0.99      0.99      0.99      3624
#          Food & Dining       0.67      1.00      0.80         4
#            Furnishings       0.97      0.97      0.97       474
#             Gas & Fuel       0.96      0.96      0.96      3627
#                   Gift       0.96      0.97      0.97      2290
#              Groceries       1.00      0.99      0.99      2689
#                    Gym       1.00      1.00      1.00      4353
#                   Hair       0.99      0.99      0.99       146
#                Hobbies       0.98      0.98      0.98       110
#       Home Improvement       0.96      0.95      0.95       891
#          Home Services       1.00      0.71      0.83         7
#          Home Supplies       0.00      0.00      0.00         0
#                  Hotel       0.94      0.91      0.92      2994
#                 Income       0.96      0.99      0.98      2097
#        Interest Income       1.00      0.86      0.92        28
#               Internet       0.81      0.81      0.81        16
#               Late Fee       0.96      0.94      0.95       272
#                Laundry       0.97      0.92      0.94       726
#          Lawn & Garden       0.98      0.98      0.98      6467
#                  Legal       1.00      0.91      0.95        43
#         Life Insurance       0.78      0.78      0.78         9
#          Misc Expenses       0.97      0.94      0.95       329
#           Mobile Phone       0.99      0.97      0.98       146
#             Motorcycle       0.94      0.90      0.92       652
#          Movies & DVDs       0.96      0.98      0.97       251
#                  Music       0.92      0.90      0.91       898
# Newspapers & Magazines       0.98      0.98      0.98      1862
#        Office Supplies       0.99      0.98      0.99      2137
#                Parking       0.99      1.00      0.99      2117
#          Personal Care       0.97      0.95      0.96      3259
#    Pet Food & Supplies       0.92      0.89      0.90       406
#               Pharmacy       0.92      1.00      0.96        57
#               Printing       0.99      0.88      0.93       716
#  Public Transportation       0.99      1.00      0.99       945
#      Rental Car & Taxi       0.80      0.77      0.79        31
#            Restaurants       0.97      0.98      0.98        64
#
#            avg / total       0.96      0.96      0.96     69791

# Size of train data(m):
#  348955
# Testing classifier with train-test split
# test size is: 10%
# Classification report:
#                         precision    recall  f1-score   support
#
#            Advertising       0.95      0.91      0.93       138
#             Air Travel       1.00      1.00      1.00      3004
#         Alcohol & Bars       1.00      0.86      0.92         7
#              Amusement       0.92      0.90      0.91        62
#                   Arts       0.97      0.97      0.97        71
#           Auto Payment       1.00      0.75      0.86         4
#               Bank Fee       1.00      1.00      1.00         1
#                Bicycle       0.93      0.98      0.96      3229
#                  Bonus       0.91      0.88      0.90      1338
#       Books & Supplies       0.91      0.91      0.91       411
#      Business Services       0.94      0.91      0.92       251
#                Charity       0.80      0.89      0.84        18
#               Clothing       1.00      1.00      1.00         4
#           Coffee Shops       1.00      0.50      0.67         2
#    Credit Card Payment       0.93      0.88      0.91       147
#               Delivery       0.95      0.89      0.92       198
#                Dentist       0.87      0.90      0.89      2631
#                 Doctor       0.91      0.83      0.87        12
#              Education       1.00      1.00      1.00         7
# Electronics & Software       0.96      0.94      0.95       452
#          Entertainment       1.00      1.00      1.00       125
#                Eyecare       0.97      0.91      0.94        43
#              Fast Food       0.89      0.92      0.90       238
#         Fees & Charges       1.00      0.95      0.97       216
#          Food & Dining       0.98      0.91      0.94       100
#            Furnishings       0.99      0.99      0.99      1781
#             Gas & Fuel       0.50      0.14      0.22         7
#                   Gift       0.97      0.96      0.97       273
#              Groceries       0.96      0.97      0.96      1785
#                    Gym       0.97      0.98      0.97      1182
#                   Hair       1.00      0.99      0.99      1313
#                Hobbies       1.00      1.00      1.00      2068
#       Home Improvement       1.00      0.99      0.99        74
#          Home Services       0.95      1.00      0.97        56
#          Home Supplies       0.96      0.95      0.95       481
#                  Hotel       1.00      0.67      0.80         6
#                 Income       1.00      1.00      1.00         4
#        Interest Income       0.95      0.91      0.93      1531
#               Internet       0.95      0.99      0.97       992
#               Late Fee       0.75      0.90      0.82        10
#                Laundry       0.86      0.75      0.80         8
#          Lawn & Garden       0.98      0.92      0.95       136
#                  Legal       0.96      0.95      0.96       380
#         Life Insurance       0.98      0.97      0.97      3269
#          Misc Expenses       1.00      0.91      0.95        33
#           Mobile Phone       0.83      0.83      0.83         6
#             Motorcycle       0.93      0.95      0.94       133
#          Movies & DVDs       0.97      0.98      0.97        57
#                  Music       0.92      0.91      0.92       331
# Newspapers & Magazines       0.96      0.97      0.97       139
#        Office Supplies       0.91      0.92      0.91       454
#                Parking       0.98      0.97      0.98       943
#          Personal Care       0.99      0.99      0.99      1042
#    Pet Food & Supplies       0.99      0.99      0.99      1045
#               Pharmacy       0.98      0.95      0.96      1611
#               Printing       0.00      0.00      0.00         1
#  Public Transportation       0.91      0.89      0.90       203
#      Rental Car & Taxi       1.00      0.90      0.95        20
#            Restaurants       0.99      0.88      0.93       354
#      Returned Purchase       1.00      0.99      0.99       479
#        Service & Parts       0.62      0.89      0.73         9
#            Service Fee       0.98      0.94      0.96        50
#               Shipping       0.00      0.00      0.00         1
#
#            avg / total       0.96      0.96      0.96     34976
#
#
# Classification report:
#                         precision    recall  f1-score   support
#
#            Advertising       1.00      0.50      0.67         8
#             Air Travel       0.00      0.00      0.00         1
#         Alcohol & Bars       1.00      1.00      1.00         2
#              Amusement       1.00      1.00      1.00         2
#                   Arts       0.00      0.00      0.00         0
#           Auto Payment       0.00      0.00      0.00         4
#               Bank Fee       0.92      0.92      0.92        25
#                Bicycle       1.00      1.00      1.00         1
#                  Bonus       1.00      0.38      0.55        16
#       Books & Supplies       1.00      1.00      1.00         1
#      Business Services       1.00      1.00      1.00         1
#                Charity       0.67      1.00      0.80         8
#               Clothing       1.00      0.33      0.50         3
#           Coffee Shops       0.00      0.00      0.00         2
#    Credit Card Payment       0.50      1.00      0.67         1
#               Delivery       0.00      0.00      0.00         1
#                Dentist       0.70      1.00      0.82         7
#                 Doctor       1.00      1.00      1.00         2
#              Education       1.00      1.00      1.00         1
# Electronics & Software       0.00      0.00      0.00         1
#          Entertainment       1.00      1.00      1.00         1
#                Eyecare       1.00      1.00      1.00         2
#              Fast Food       1.00      1.00      1.00         1
#         Fees & Charges       1.00      1.00      1.00         1
#          Food & Dining       1.00      1.00      1.00         3
#            Furnishings       1.00      1.00      1.00         3
#             Gas & Fuel       1.00      1.00      1.00         4
#                   Gift       0.00      0.00      0.00         0
#              Groceries       1.00      1.00      1.00         1
#                    Gym       1.00      1.00      1.00         3
#                   Hair       0.00      0.00      0.00         1
#                Hobbies       0.82      1.00      0.90        45
#       Home Improvement       0.78      0.84      0.81        25
#          Home Services       0.25      1.00      0.40         1
#          Home Supplies       0.80      0.73      0.76        11
#                  Hotel       1.00      1.00      1.00         1
#                 Income       1.00      0.75      0.86         4
#        Interest Income       1.00      1.00      1.00         6
#
#            avg / total       0.83      0.82      0.80       200
