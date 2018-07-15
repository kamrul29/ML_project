#import nltk
#nltk.download()
from sklearn import datasets
from sklearn import svm
import pandas as pd
from numpy import genfromtxt
# Stemming purposes
from nltk.stem import PorterStemmer
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import sent_tokenize, word_tokenize

# Splitting Dataset
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

from sklearn import metrics
import numpy as np
import itertools
#Helper

import matplotlib.pyplot as plt


#PLOT Matrix

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
   
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('')
#         print('Confusion matrix, with little normalization')

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()


# <center>  Dataset Parse </center>



data_set = pd.read_csv("fake_or_real_news.csv")



# <center> Stemming Approach </center>



stemmer = SnowballStemmer("english", ignore_stopwords=True)
def stemmer_function(text):
    words = word_tokenize(text)
    ret = "";
    for w in words:
        ret = ret + " " + stemmer.stem(w)
    return ret
# print(stemmer_function("I  am eating rice going went gone"))


# <center> Stemming Text </center>



news = data_set['text']
y = data_set.label 
index_array = y.index.values
verdict = []
stemmed_news = []
counter = 0
for index in index_array:
    val = y.at[index]
    verdict.append(val)
    n = stemmer_function(news.at[index])
    stemmed_news.append(n)
    counter += 1
    




X_train, X_test, y_train, y_test = train_test_split(stemmed_news, verdict, test_size=0.4, random_state=53)
print("Dataset Splitted.")
# print(stemmed_news[1:12])


    


# <center>  Applying TF-IDF </center>



count_vectorizer = CountVectorizer(stop_words='english')
count_train = count_vectorizer.fit_transform(X_train) 
count_test = count_vectorizer.transform(X_test)

tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7) 
tfidf_train = tfidf_vectorizer.fit_transform(X_train) 
tfidf_test = tfidf_vectorizer.transform(X_test)

print("TF-IDF Training Started\n->->->")
# print(tfidf_train)
print("TF-IDF Training Ended")




# <center>  Feature Extract </center>



print('Sample features names\n',tfidf_vectorizer.get_feature_names()[1500:1600:5], '\n')
# print(count_vectorizer.get_feature_names()[900:910])

feature_arr = tfidf_vectorizer.get_feature_names()

for i in range(1500, 1600, 5):
    val = tfidf_vectorizer.vocabulary_[feature_arr[i]]
    print(feature_arr[i] , " ----> " , val)
# print(tfidf_vectorizer)





# <center>  SVM Classifier </center>



from sklearn import svm

clf_svm = svm.SVC(probability=True, C=1000)

clf_svm.fit(tfidf_train, y_train)
pred_svm = clf_svm.predict(tfidf_test)
score = metrics.accuracy_score(y_test, pred_svm)
print("Accuracy:   %0.2f" % (score*100.) + "%")
cm = metrics.confusion_matrix(y_test, pred_svm, labels=['FAKE', 'REAL'])
plot_confusion_matrix(cm, 
                      classes=['FAKE', 'REAL'], 
                      title="Confusion Matrix for \nNaive-Bayes Classifier",
                      cmap = plt.cm.Greys
                     )


    #Accuracy:   81.49%
    



#![png](output_20_1.png)

