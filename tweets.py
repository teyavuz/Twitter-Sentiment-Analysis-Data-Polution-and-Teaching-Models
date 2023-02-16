import snscrape.modules.twitter as sntwitter
import pandas as pd
import re,string
import numpy as np
import nltk
import matplotlib.pyplot as plt


columns=['tweet','label','sorgu tipi','tweetUrl','like','user','date']
data=[]

#https://gist.github.com/slowkow/7a7f61f495e3dbb7e3d767f97bd7304b
def clean_text(text):
    tweet = re.sub("@[A-Za-z0-9]+","",text.lower())
    tweet = re.sub(r"(?:\@|http?\://|https?\://|www)\S+", "", tweet) 
    tweet = re.sub("&lt;/?[a-z]+&gt;", "", tweet) 
    tweet = re.sub("&amp;", "", tweet) 
    tweet = " ".join(tweet.split())
   # tweet = remove_emoji(tweet) 
    tweet = tweet.replace("#", "").replace("_", " ") 
    emoji_pattern = re.compile("["
                           u"\U0001F600-\U0001F64F"  # emoticons
                               u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                               u"\U0001F680-\U0001F6FF"  # transport & map symbols
                               u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                               u"\U00002500-\U00002BEF"  # chinese char
                               u"\U00002702-\U000027B0"
                               u"\U00002702-\U000027B0"
                               u"\U000024C2-\U0001F251"
                               u"\U0001f926-\U0001f937"
                               u"\U00010000-\U0010ffff"
                               u"\u2640-\u2642"
                               u"\u2600-\u2B55"
                               u"\u200d"
                               u"\u23cf"
                               u"\u23e9"
                               u"\u231a"
                               u"\ufe0f"  # dingbats
                               u"\u3030"
                           "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', tweet)


maxTweets=2000
query = '"vindiesel" "vin diesel" lang:en -filter:links -filter:replies"'


def scrap_tweet(hashtag,maxTweet):
    count=1
    for i,tweet in enumerate(sntwitter.TwitterHashtagScraper(query).get_items()):
        if i>maxTweet:
            break
        data.append([clean_text(tweet.content),'',query,tweet.url,tweet.likeCount,tweet.user,tweet.date])
        count+=1
        
scrap_tweet(query,maxTweets)


d=pd.DataFrame(data,columns=columns)

d.drop_duplicates(subset=['tweet'])

print(d.head)

d.to_csv('tweet.csv',encoding='utf-8-sig')