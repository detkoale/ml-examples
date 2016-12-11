from sklearn import datasets
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC

newsgroups = datasets.fetch_20newsgroups(
                    subset='all',
                    categories=['alt.atheism', 'sci.space']
             )

textToDigits = TfidfVectorizer()
textToDigits.fit_transform(newsgroups.data)

clf = SVC()
clf.coef_.conjugate()