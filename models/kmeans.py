#kümeleme

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics



style.use('ggplot')
df=pd.read_csv('datatweet.csv')

df.head()

df.info()

tweet=df['tweet']

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(tweet)
print(X.toarray())

k=3
km= KMeans(n_clusters=k,init='k-means++',random_state=42)
km.fit(X)

y=km.labels_
df['sentiment']=km.labels_
predict=km.predict(X)
predict

df['sentiment'].value_counts()

fig =plt.figure(figsize=(5,5))
sns.countplot(x='sentiment',data=df)

#https://www.youtube.com/watch?v=B0BkUNMqfSo grafiğ,n videosu
pca=PCA(n_components=3)
scatter_plot_points=pca.fit_transform(X.toarray())

colors=["r","g","b"]
x_axis=[o[0] for o in scatter_plot_points]
y_axis=[o[1] for o in scatter_plot_points]

fig,ax=plt.subplots(figsize=(20,20))
ax.scatter(x_axis, y_axis,c=[colors[d] for d in predict])

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)

print("trainign data shape:",X_train.shape)
print("testing data shape:",X_test.shape)

knn_model=KNeighborsClassifier(n_neighbors=3)
knn_model.fit(X_train,y_train)

predictions=knn_model.predict(X_test)

print(metrics.confusion_matrix(y_test,predictions))

print(metrics.classification_report(y_test,predictions))

print(metrics.accuracy_score(y_test,predictions))

print(knn_model.score(X_train,y_train))