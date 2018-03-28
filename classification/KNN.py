import pandas as pd
import re
import math
import numpy as np
from sklearn.model_selection import KFold
from sklearn import neighbors
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import preprocessing
from sklearn import metrics
from termcolor import colored

def trainTFIDF(texts, labels):
    transformer = TfidfVectorizer()
    trainTexts = transformer.fit_transform(texts)
    classifier = neighbors.KNeighborsClassifier(5)
    classifier.fit(trainTexts, labels)

    print("train n_samples: %d, n_features: %d" % trainTexts.shape)
    if len(transformer.stop_words_):
        print("idf stop words: ")
        print(" ".join(transformer.stop_words_))

    return (classifier, transformer)

def crossValidation(texts, labels, testSampleSize=350, replication=10):
    nSplits = math.floor(len(texts) / testSampleSize)
    sum = 0
    kf = KFold(n_splits=nSplits, shuffle=True, random_state=42)
    texts = np.array(texts)
    count = 0
    rCount = replication
    if rCount >= nSplits: rCount = nSplits
    for trainIndex, testIndex in kf.split(texts):
        count += 1
        classifier, transformer = trainTFIDF(texts[trainIndex], labels[trainIndex])
        testTexts = transformer.transform(texts[testIndex])
        pred = classifier.predict(testTexts)
        score = metrics.accuracy_score(labels[testIndex], pred)
        print("test sample size: %d, accuracy: %0.3f" % (testTexts.shape[0], score))
        sum += score
        if count == rCount:
            classificationReport(labels[testIndex], pred, title='Cross-Validation report:')
            break
    average = sum / rCount
    print(f'Accuracy of KNN classification with tfidf: {average}\n')

def classificationReport(labels, predictions, title='Classification report:'):
    print(colored(title, 'magenta', attrs=['bold']))
    print(metrics.classification_report(labels, predictions, target_names=list(le.classes_)))

if __name__ == '__main__':
    trainData = pd.read_csv('./assets/res_purchase_card_cleaned.csv', sep=",")
    testData = pd.read_csv('./assets/statementsLabeled.csv', sep=",")

    print("Size of train data(m):\n", trainData.shape[0])

    le = preprocessing.LabelEncoder()

    trainItems = pd.DataFrame(trainData, columns=['Agency Name', 'Description', 'Vendor', 'category_draft_1'])
    trainTexts = trainItems["Agency Name"] + " " + trainItems["Description"] + " " + trainItems["Vendor"]
    trainLabels = le.fit_transform(trainData['category_draft_1'])

    testItems = pd.DataFrame(testData, columns=['TITLE', 'Category Draft-1'])
    testTexts = testItems['TITLE']
    testlabels = le.fit_transform(testData['Category Draft-1'])

    trainTexts = [re.sub(r'([^a-zA-Z0-9])+', ' ', s) for s in trainTexts]
    trainTexts = [re.sub(r'(\s)+', ' ', s) for s in trainTexts]

    testTexts = [re.sub(r'([^a-zA-Z0-9])+', ' ', s) for s in testTexts]
    testTexts = [re.sub(r'(\s)+', ' ', s) for s in testTexts]

    classifier, transformer = trainTFIDF(trainTexts, trainLabels)
    testTexts = transformer.transform(testTexts)
    predLabels = classifier.predict(testTexts)
    score = metrics.accuracy_score(testlabels, predLabels)
    print("test sample size: %d, accuracy: %0.3f" % (testTexts.shape[0], score))

    print("\nCross validation:\n")
    crossValidation(trainTexts, trainLabels)

    classificationReport(testlabels, predLabels)
