import sklearn
import sklearn.feature_extraction.text


def bagOfWords(files_data):
    """
    Converts a list of strings (which are loaded from files) to a BOW representation of it
    parameter 'files_data' is a list of strings
    returns a `scipy.sparse.coo_matrix`
    """

    count_vector = sklearn.feature_extraction.text.CountVectorizer()
    return count_vector.fit_transform(files_data)
