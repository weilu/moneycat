import pandas as pd
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

    # print("Number of Columns:\n", origin_data.shape[1], "\n\n")
    # print("List of Columns:\n", ", ".join(origin_data.columns), "\n\n")
    # print("Data:\n", origin_data.head(), "\n\n")
    print("Size of train data(m):\n", origin_data.shape[0])

    transaction_info = pd.DataFrame(origin_data, columns=['Description', 'Vendor', 'category_draft_1'])

    interactions_text = transaction_info["Description"] + " " + transaction_info["Vendor"]
    interactions_label = transaction_info["category_draft_1"].astype('category').cat.codes
    y_names = dict(enumerate(transaction_info["category_draft_1"].astype('category').cat.categories)).values()

    # calculate the BOW representation
    print(colored('Calculating BOW', 'green', attrs=['bold']))

    word_counts = util.bagOfWords(interactions_text)

    # TFIDF
    print(colored('Calculating TFIDF', 'green', attrs=['bold']))
    tf_transformer = sklearn.feature_extraction.text.TfidfTransformer(use_idf=True).fit(word_counts)
    X = tf_transformer.transform(word_counts)

    # create classifier
    if METHOD == "MNB":
        clf = sklearn.naive_bayes.MultinomialNB()
    elif METHOD == "SVM":
        clf = sklearn.svm.LinearSVC()
    elif METHOD == "KNN":
        n_neighbors = 11
        weights = 'uniform'
        # weights = 'distance'
        clf = sklearn.neighbors.KNeighborsClassifier(n_neighbors, weights=weights)

    # test the classifier
    print(colored('Testing classifier with train-test split', 'magenta', attrs=['bold']))
    test_classifier(X, interactions_label, clf, test_size=0.2, y_names=y_names, confusion=False)


def test_classifier(X, y, clf, test_size=0.4, y_names=None, confusion=False):
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


if __name__ == '__main__':
    main()

# KNN & MNB
# Memory Error

# SVM
# Classification report:
#                         precision    recall  f1-score   support
#
#            Advertising       0.93      0.90      0.91       296
#             Air Travel       1.00      1.00      1.00      5995
#         Alcohol & Bars       1.00      0.33      0.50         3
#              Amusement       0.88      0.83      0.86        18
#                   Arts       0.94      0.80      0.87       122
#           Auto Payment       0.97      0.89      0.93       117
#                Bicycle       1.00      0.90      0.95        20
#       Books & Supplies       0.94      0.97      0.95      6324
#      Business Services       0.87      0.92      0.90      2626
#                Charity       0.92      0.90      0.91       775
#               Clothing       0.93      0.89      0.91       537
#           Coffee Shops       1.00      1.00      1.00         3
#                 Doctor       0.90      0.83      0.86       274
#              Education       0.92      0.91      0.92       405
# Electronics & Software       0.87      0.89      0.88      5170
#          Entertainment       0.79      0.77      0.78        35
#                Eyecare       1.00      1.00      1.00         5
#              Fast Food       0.97      0.94      0.96       812
#         Fees & Charges       1.00      1.00      1.00       232
#          Food & Dining       0.97      0.93      0.95        82
#            Furnishings       0.87      0.89      0.88       448
#             Gas & Fuel       0.99      0.96      0.98       384
#                   Gift       0.96      0.91      0.93       214
#              Groceries       0.99      0.99      0.99      3633
#                   Hair       1.00      1.00      1.00         2
#                Hobbies       0.98      0.94      0.96       524
#       Home Improvement       0.96      0.97      0.96      3708
#          Home Services       0.97      0.97      0.97      2318
#          Home Supplies       1.00      0.99      0.99      2562
#                  Hotel       1.00      1.00      1.00      4299
#               Internet       0.99      0.97      0.98       150
#                Laundry       0.96      1.00      0.98       129
#          Lawn & Garden       0.96      0.94      0.95       942
#                  Legal       1.00      0.67      0.80         6
#          Misc Expenses       0.94      0.91      0.93      3003
#           Mobile Phone       0.96      0.99      0.98      2142
#             Motorcycle       0.91      0.80      0.85        25
#          Movies & DVDs       0.88      0.81      0.84        26
#                  Music       0.94      0.96      0.95       215
# Newspapers & Magazines       0.94      0.94      0.94       693
#        Office Supplies       0.98      0.97      0.98      6523
#                Parking       0.94      0.92      0.93        51
#          Personal Care       0.90      0.69      0.78        13
#    Pet Food & Supplies       0.95      0.95      0.95       286
#               Pharmacy       0.99      0.98      0.98       140
#               Printing       0.94      0.91      0.92       662
#      Rental Car & Taxi       0.99      0.97      0.98       260
#            Restaurants       0.94      0.91      0.92       867
#        Service & Parts       0.98      0.97      0.98      1918
#            Service Fee       0.99      0.99      0.99      2030
#               Shipping       0.99      1.00      1.00      2085
#               Shopping       0.97      0.94      0.96      3370
#          Spa & Massage       1.00      0.75      0.86         4
#         Sporting Goods       0.94      0.87      0.90       418
#                Storage       1.00      0.93      0.96        55
#             Television       0.98      0.89      0.94       761
#              Utilities       0.99      0.99      0.99       965
#               Vacation       0.85      0.79      0.82        43
#             Veterinary       0.86      1.00      0.92        66
#
#            avg / total       0.96      0.96      0.96     69791
