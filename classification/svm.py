import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

origin_data = pd.read_csv('./assets/res_purchase_card_cleaned.csv', sep=",", error_bad_lines=False)

# print("Number of Columns:\n", origin_data.shape[1], "\n\n")
# print("List of Columns:\n", ", ".join(origin_data.columns), "\n\n")
# print("Data:\n", origin_data.head(), "\n\n")
print("Size of train data(m):\n", origin_data.shape[0])

transaction_info = pd.DataFrame(origin_data, columns=['Description', 'Vendor', 'category_draft_1'])

interactions_train, interactions_test = train_test_split(transaction_info,
                                                         stratify=transaction_info['category_draft_1'],
                                                         test_size=0.20,
                                                         random_state=42)

train_text = interactions_train["Description"] + " " + interactions_train["Vendor"]
train_label = interactions_train["Description"]

test_text = interactions_test["Description"] + " " + interactions_test["Vendor"]
test_label = interactions_test["Description"]

text_clf = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()), ('clf', MultinomialNB()), ])

text_clf = text_clf.fit(train_text, train_label)

predicted = text_clf.predict(test_text)
print(np.mean(predicted == text_clf))
#
# vectorizer = TfidfVectorizer(analyzer='word',
#                              ngram_range=(1, 2),
#                              min_df=0.003,
#                              max_df=0.5,
#                              max_features=5000
#                              )
#
# tfidf_matrix = vectorizer.fit_transform(train_text)
# print(tfidf_matrix)
