import pandas as pd
import matplotlib.pyplot as plt
from textblob import TextBlob
import numpy as np
import seaborn as sns


dataSource = 'tweet.csv'
tweets = pd.read_csv(dataSource)

lambda x:   TextBlob(x).polarity == 0

tweets['polarity'] = tweets.tweet.apply(lambda x: TextBlob(x).polarity)
tweets['subjectivity'] = tweets.tweet.apply(lambda x: TextBlob(x).subjectivity)
tweets['sentiment'] = np.where(tweets.polarity > 0, 'positive', np.where(tweets.polarity < 0, 'negative', 'neutral'))

tweets.to_csv('tweet.csv')
print(tweets.head())

#tweets['sentiment'] = np.where(tweets.polarity > 0, 'positive', np.where(tweets.polarity < 0, 'negative', 'neutral'))
print(tweets.head())


# Polarity oranı en yüksek olan 5 tweet
yuksekpolarity = tweets.nlargest(5,'polarity')['tweet']

# Polarity oranı en düşük 5 tweet
dusukpolarity = tweets.nsmallest(5,'polarity')['tweet']


print(tweets['sentiment'].value_counts())




num_bins = 50
plt.figure(figsize=(10,6))
n, bins, patches = plt.hist(tweets.polarity, num_bins, facecolor='purple', alpha=0.5)
plt.xlabel('Kirlilik Dagilimi')
plt.ylabel('Tweet Adedi')
plt.title('Kirlilik histogrami')
plt.show()

plt.figure(figsize=(10,6))
sns.boxenplot(x='subjectivity', y='polarity', data=tweets)
plt.show()

plt.figure(figsize=(15,5))
plt.title('Kirlilik ve Subjektiv histogrami',fontsize=12,fontweight='bold')
sns.kdeplot(tweets['polarity'], label='Kirlilik Dagilimi', lw=2.5)
sns.kdeplot(tweets['subjectivity'], label='Subjektiv Dagilimi', lw=2.5)
plt.xlabel('Polarity|subjetivity Değeri', fontsize=10)
plt.ylabel('Aralık', fontsize=10)
# Display the generated image:

plt.legend()
plt.show()