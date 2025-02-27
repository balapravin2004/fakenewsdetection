import pandas as pd
from sklearn.model_selection import train_test_split
import sklearn
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix
from matplotlib import pyplot as plt
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.model_selection import KFold
import itertools
import numpy as np
import seaborn as sb
import pickle
import os 

df = pd.read_csv("news.csv")

def create_distribution(dataFile):
    return sb.countplot(x='label', data=dataFile, palette='hls')

create_distribution(df)

def data_qualityCheck():
    # print("Checking data qualitites...")
    df.isnull().sum()
    df.info()  
    # print("check finished.")
data_qualityCheck()

y = df.label
# y.head()

df.drop("label", axis=1)

X_train, X_test, y_train, y_test = train_test_split(df['text'], y, test_size=0.33, random_state=53)

count_vectorizer = CountVectorizer(stop_words='english')

count_train = count_vectorizer.fit_transform(X_train) 

# def get_countVectorizer_stats():
    
    
#     print(count_train.shape)

    
#     print(count_vectorizer.vocabulary_)

# get_countVectorizer_stats()

count_test = count_vectorizer.transform(X_test)

tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)

tfidf_train = tfidf_vectorizer.fit_transform(X_train) 

def get_tfidf_stats():
    tfidf_train.shape
    
    # print(tfidf_train.A[:10])

get_tfidf_stats()

tfidf_test = tfidf_vectorizer.transform(X_test)

# print(tfidf_vectorizer.get_feature_names_out()[-10:])

count_df = pd.DataFrame(count_train.A, columns=count_vectorizer.get_feature_names_out())
tfidf_df = pd.DataFrame(tfidf_train.A, columns=tfidf_vectorizer.get_feature_names_out())
difference = set(count_df.columns) - set(tfidf_df.columns)
# print(difference)

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
        # print("Normalized confusion matrix")        #useful
    else:
        # print('Confusion matrix, without normalization')        #useful

        thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


nb_pipeline = Pipeline([
        ('NBTV',tfidf_vectorizer),
        ('nb_clf',MultinomialNB())])

nb_pipeline.fit(X_train,y_train)

predicted_nbt = nb_pipeline.predict(X_test)

score = metrics.accuracy_score(y_test, predicted_nbt)
# print(f'Accuracy: {round(score*100,2)}%')

cm = metrics.confusion_matrix(y_test, predicted_nbt, labels=['FAKE', 'REAL'])
plot_confusion_matrix(cm, classes=['FAKE', 'REAL'])

nbc_pipeline = Pipeline([
        ('NBCV',count_vectorizer),
        ('nb_clf',MultinomialNB())])
nbc_pipeline.fit(X_train,y_train)

predicted_nbc = nbc_pipeline.predict(X_test)
score = metrics.accuracy_score(y_test, predicted_nbc)
# print(f'Accuracy: {round(score*100,2)}%')

cm1 = metrics.confusion_matrix(y_test, predicted_nbc, labels=['FAKE', 'REAL'])
plot_confusion_matrix(cm1, classes=['FAKE', 'REAL'])

linear_clf = Pipeline([
    ('tfidf', tfidf_vectorizer),  # Use tfidf_vectorizer for vectorization
    ('pa_clf', PassiveAggressiveClassifier(max_iter=50))  # PassiveAggressiveClassifier as the estimator
])

linear_clf.fit(X_train, y_train)

pred = linear_clf.predict(X_test)
score = metrics.accuracy_score(y_test, pred)
# print(f'Accuracy: {round(score*100,2)}%')       #useful

cm = metrics.confusion_matrix(y_test, pred, labels=['FAKE', 'REAL'])
plot_confusion_matrix(cm, classes=['FAKE', 'REAL'])

# Save the trained model using pickle
model_file = 'final_model.sav'
pickle.dump(linear_clf, open(model_file, 'wb'))

