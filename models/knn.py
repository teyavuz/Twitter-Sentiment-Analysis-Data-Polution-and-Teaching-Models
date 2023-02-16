import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics





df=pd.read_csv('datatweet.csv')
#df=pd.read_csv('tweet.csv')

df.head()

tweet=df['tweet']
y=df['sentiment'].values

print(tweet.head())

vectorizer = TfidfVectorizer()
X=vectorizer.fit_transform(tweet)

print(X.toarray())

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0,shuffle=True)

print("trainign data shape:",X_train.shape)
print("testing data shape:",X_test.shape)

knn_model=KNeighborsClassifier(n_neighbors=3)
knn_model.fit(X_train,y_train)

predictions=knn_model.predict(X_test)
print(metrics.confusion_matrix(y_test,predictions))

print(metrics.classification_report(y_test,predictions))

print(metrics.accuracy_score(y_test,predictions))

print(knn_model.score(X_train,y_train))

sent_test_vector=vectorizer.transform(["****"])
print(sent_test_vector)

y_pred=knn_model.predict(sent_test_vector.toarray())
print(y_pred)