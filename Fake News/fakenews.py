#import nltk
#nltk.download()
#new pc te run korate hole nltk library download kora lagbe
#tokhon eta uncomment kore download korte hbe,then nltk import hoye snowballstemmer kaj korbe
from sklearn import datasets
from sklearn import svm
import pandas as pd
from numpy import genfromtxt

from nltk.stem import PorterStemmer
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import sent_tokenize, word_tokenize


from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

from sklearn import metrics
import numpy as np
import itertools

#here we will make a function to draw confusion matrix

import matplotlib.pyplot as plt


#PLOT Matrix
#confusion matrix building
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


    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()





#taking input the data

data_set = pd.read_csv("fake_or_real_news.csv")


#preprocessing the data with semantic approach




stemmer = SnowballStemmer("english", ignore_stopwords=True)
def stemmer_function(text):
    words = word_tokenize(text)
    ret = "";
    for w in words:
        ret = ret + " " + stemmer.stem(w)
    return ret



# making Stemming Text 



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
    '''if counter % 1000 == 0:
        print('Number of data being processed step by step: ', counter , "of", len(news), str(counter*1./len(news)*100.0) + '%')'''




X_train, X_test, y_train, y_test = train_test_split(stemmed_news, verdict, test_size=0.3, random_state=53)

# print(stemmed_news[1:12])


    


# now are applying Applying TF-IDF  processing



count_vectorizer = CountVectorizer(stop_words='english')
count_train = count_vectorizer.fit_transform(X_train) 
count_test = count_vectorizer.transform(X_test)

tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7) 
tfidf_train = tfidf_vectorizer.fit_transform(X_train) 
tfidf_test = tfidf_vectorizer.transform(X_test)





# Now traning the dataset with Random Forest Classifier 


print(" Now traning the dataset with Random Forest Classifier")
from sklearn.ensemble import RandomForestClassifier

clf_rand = RandomForestClassifier(n_estimators = 26 , criterion = 'entropy' , random_state = 0)

clf_rand.fit(tfidf_train, y_train)

pred_rand = clf_rand.predict(tfidf_test)

score = metrics.accuracy_score(y_test, pred_rand)

print("Accuracy of random forest:   %0.2f" % (score*100.) + "%")
cm = metrics.confusion_matrix(y_test, pred_rand, labels=['FAKE', 'REAL'])
plot_confusion_matrix(cm, 
                      classes=['FAKE', 'REAL'], 
                      title="Confusion Matrix for \nRandom Forest Classifier",
                      cmap = plt.cm.Greens
                     )


    
    






# Now traning the dataset with Naive Bayes </center>


print(" Now traning the dataset with Naive Bayes Classifier")
from sklearn.naive_bayes import MultinomialNB
import itertools

naive_classifier = MultinomialNB()
naive_classifier.fit(tfidf_train, y_train)
pred_tree = naive_classifier.predict(tfidf_test)

score = metrics.accuracy_score(y_test, pred_tree)
print("Accuracy of native bayes:   %0.2f" % (score*100.) + "%")
cm = metrics.confusion_matrix(y_test, pred_tree, labels=['FAKE', 'REAL'])
plot_confusion_matrix(cm, 
                      classes=['FAKE', 'REAL'], 
                      title="Confusion Matrix for \nNaive-Bayes Classifier",
                      cmap = plt.cm.Reds
                     )


    
    






# Now traning the dataset with SVM Classifier


print("Now traning the dataset with SVM Classifier")
from sklearn import svm

clf_svm = svm.SVC(probability=True, C=1000)

clf_svm.fit(tfidf_train, y_train)
pred_svm = clf_svm.predict(tfidf_test)
score = metrics.accuracy_score(y_test, pred_svm)
print("Accuracy with svm:   %0.2f" % (score*100.) + "%")
cm = metrics.confusion_matrix(y_test, pred_svm, labels=['FAKE', 'REAL'])
plot_confusion_matrix(cm, 
                      classes=['FAKE', 'REAL'], 
                      title="Confusion Matrix for SVM Classifier",
                      cmap = plt.cm.Greys
                     )


    
    





