#Gauss Modeli

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
df=pd.read_csv('datatweet.csv')
#df=pd.pandas.read_excel('ws.xlsx')
#df=pd.pandas.read_excel('x.xlsx')
df.info()


print(df.head())

tweet=[doc for doc in df['tweet']]
kirlilik=[doc for doc in df['sentiment']]


vectorizer = TfidfVectorizer(analyzer='word')
sen_tranin_vector=vectorizer.fit_transform(tweet)

#print(sen_tranin_vector.toarray())

from sklearn.naive_bayes import GaussianNB
clf=GaussianNB()
model=clf.fit(X=sen_tranin_vector.toarray(),y=kirlilik)

predictions=model.predict(sen_tranin_vector.toarray())
from sklearn import metrics
#print(metrics.confusion_matrix(kirlilik,predictions))

#print(metrics.classification_report(kirlilik,predictions))

print(metrics.accuracy_score(kirlilik,predictions))