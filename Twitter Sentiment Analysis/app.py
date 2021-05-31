'''This is main python flask application file , programme execution starts from here'''

# importing all dependencies
from flask import Flask,render_template,request
import requests

import nltk
import numpy as np
import re
import pandas as pd
import pylab as pl
import matplotlib.pyplot as plt

from nltk.tokenize import WordPunctTokenizer
#from bs4 import BeautifulSoup
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA, TruncatedSVD

from sklearn import metrics
from mpl_toolkits.mplot3d import Axes3D

from matplotlib import pyplot

#plt.style.use('fivethirtyeight')
#%matplotlib inline
#%config InlineBackend.figure_format = 'retina'

#!pip install vaderSentiment
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

analyser = SentimentIntensityAnalyzer()

#!pip install datashader
import datashader as ds
import datashader.transfer_functions as tf


from tweepy import API
from tweepy import Cursor
from tweepy.streaming import StreamListener
from tweepy import OAuthHandler
from tweepy import Stream


import tweepy


import twitter_credentials
import numpy as np
import pandas as pd


from nltk.stem.porter import *
from textblob import TextBlob
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.cluster import KMeans


# creating a Flask app and name it 'app'
app = Flask(__name__)



class TwitterClient():
    def __init__(self, twitter_user=None):
        self.auth = TwitterAuthenticator().authenticate_twitter_app()
        self.twitter_client = API(self.auth)

        self.twitter_user = twitter_user

    def get_twitter_client_api(self):
        return self.twitter_client


class TwitterAuthenticator():

    def authenticate_twitter_app(self):
        auth = OAuthHandler(twitter_credentials.CONSUMER_KEY, twitter_credentials.CONSUMER_SECRET)
        auth.set_access_token(twitter_credentials.ACCESS_TOKEN, twitter_credentials.ACCESS_TOKEN_SECRET)
        return auth


class TwitterStreamer():

    def __init__(self):
        self.twitter_autenticator = TwitterAuthenticator()

    def stream_tweets(self, hash_tag_list):

        listener = TwitterListener()
        auth = self.twitter_autenticator.authenticate_twitter_app()
        stream = Stream(auth, listener)


        stream.filter(track=hash_tag_list)

class TweetAnalyzer():

    def tweets_to_data_frame(self, tweets):
        #for tweet in tweets:
            #print(tweet.full_text)

        df = pd.DataFrame(data=[tweet.full_text for tweet in tweets], columns=['Tweets'])

        df['id'] = np.array([tweet.id for tweet in tweets])
        df['len'] = np.array([len(tweet.full_text) for tweet in tweets])
        df['date'] = np.array([tweet.created_at for tweet in tweets])
        df['source'] = np.array([tweet.source for tweet in tweets])
        df['likes'] = np.array([tweet.favorite_count for tweet in tweets])
        df['retweets'] = np.array([tweet.retweet_count for tweet in tweets])

        return df

def remove_pattern(input_txt, pattern):
    r = re.findall(pattern, input_txt)
    for i in r:
        input_txt = re.sub(i, '', input_txt)

    return input_txt

def calculate_sentiment(Clean_text):
    return TextBlob(Clean_text).sentiment

def calculate_sentiment_analyser(Clean_text):
    return analyser.polarity_scores(Clean_text)


def calculate_sentiment(Tweet):
    return TextBlob(Tweet).sentiment

def calculate_sentiment_analyser(Tweet):
    return analyser.polarity_scores(Tweet)

# default ('/') route of the application
@app.route('/',methods=['GET','POST'])
def home():
    # this part executes when users make a POST request to the server
    data={}
    if request.method == 'POST':
        try:


            twitter_client = TwitterClient()
            tweet_analyzer = TweetAnalyzer()

            api = twitter_client.get_twitter_client_api()

            tweets = tweepy.Cursor(api.search, q=['Covishield','Covaxin'], lang="en",
                                       tweet_mode='extended').items(50)
            list_tweets = [tweet for tweet in tweets]

                #print(dir(tweets[0]))
                #print(tweets[0].retweet_count)

            df = tweet_analyzer.tweets_to_data_frame(list_tweets)

            print(df.head(10))
                #print(df['Tweets'])




            df['Clean_text'] = np.vectorize(remove_pattern)(df['Tweets'], "@[\w]*")

            df['Clean_text'] = df['Clean_text'].str.replace("[^a-zA-Z#]", " ")

            df['Clean_text'] = df['Clean_text'].apply(lambda x: ' '.join([w for w in x.split() if len(w)>3]))

            tokenized_tweet = df['Clean_text'].apply(lambda x: x.split())


            stemmer = PorterStemmer()

            tokenized_tweet = tokenized_tweet.apply(lambda x: [stemmer.stem(i) for i in x]) # stemming
            tokenized_tweet.head()

            for i in range(len(tokenized_tweet)):
                tokenized_tweet[i] = ' '.join(tokenized_tweet[i])

            df['Clean_text'] = tokenized_tweet

            df.loc[:,('Tweets','Clean_text')]

            unique_clean_text = df.Clean_text.unique()
            unique_full_text = df.Tweets.unique()
            print(len(unique_clean_text))
            print(len(unique_full_text))
            print(len(df))

            df.drop_duplicates(subset=['Clean_text'], keep = 'first',inplace= True)

            df.reset_index(drop=True,inplace=True)

            df['Clean_text_length'] = df['Clean_text'].apply(len)
            df.head()


            df['sentiment']=df.Clean_text.apply(calculate_sentiment)
            df['sentiment_analyser']=df.Clean_text.apply(calculate_sentiment_analyser)


            s = pd.DataFrame(index = range(0,len(df)),columns= ['compound_score','compound_score_sentiment'])

            for i in range(0,len(df)):
              s['compound_score'][i] = df['sentiment_analyser'][i]['compound']

              if (df['sentiment_analyser'][i]['compound'] <= -0.05):
                s['compound_score_sentiment'][i] = 'Negative'
              if (df['sentiment_analyser'][i]['compound'] >= 0.05):
                s['compound_score_sentiment'][i] = 'Positive'
              if ((df['sentiment_analyser'][i]['compound'] >= -0.05) & (df['sentiment_analyser'][i]['compound'] <= 0.05)):
                s['compound_score_sentiment'][i] = 'Neutral'

            df['compound_score'] = s['compound_score']
            df['compound_score_sentiment'] = s['compound_score_sentiment']
            df.head(4)

            df.compound_score_sentiment.value_counts()

            c0,d0,e0=df.compound_score_sentiment.value_counts()
            print(c0,d0,e0)

            df['Clean_text'].head()

            #tf = TfidfVectorizer(analyzer='word', ngram_range=(1, 3), min_df=0, stop_words='english')

            # Considering 3 grams and mimnimum frq as 0
            tf_idf_vect = CountVectorizer(analyzer='word',ngram_range=(1,1),stop_words='english', min_df = 0.0001)
            tf_idf_vect.fit(df['Clean_text'])
            desc_matrix = tf_idf_vect.transform(df["Clean_text"])


            num_clusters = 3
            km = KMeans(n_clusters=num_clusters,algorithm='auto', copy_x=True, init='random', #max_iter=600,
                 #n_init=10, n_jobs=1, precompute_distances='auto',
                verbose=0, n_jobs=1 ,random_state=None) #tol=0.0001, verbose=0)
            km.fit(desc_matrix)
            clusters = km.labels_.tolist()

            # create DataFrame films from all of the input files.
            tweets = {'Tweet': df["Clean_text"].tolist(), 'Cluster': clusters}
            frame = pd.DataFrame(tweets)
            frame

            frame['Cluster'].value_counts()

            neu,pos,neg = frame['Cluster'].value_counts()

            neutral_n,positive_n,negative_n = frame['Cluster'].value_counts().index
            print(neutral_n,positive_n,negative_n)



            frame['sentiment']=frame.Tweet.apply(calculate_sentiment)
            frame['sentiment_analyser']=frame.Tweet.apply(calculate_sentiment_analyser)


            s = pd.DataFrame(index = range(0,len(frame)),columns= ['compound_score','compound_score_sentiment'])

            for i in range(0,len(frame)):
              s['compound_score'][i] = frame['sentiment_analyser'][i]['compound']

              if (frame['sentiment_analyser'][i]['compound'] <= -0.05):
                s['compound_score_sentiment'][i] = 'Negative'
              if (frame['sentiment_analyser'][i]['compound'] >= 0.05):
                s['compound_score_sentiment'][i] = 'Positive'
              if ((frame['sentiment_analyser'][i]['compound'] >= -0.05) & (frame['sentiment_analyser'][i]['compound'] <= 0.05)):
                s['compound_score_sentiment'][i] = 'Neutral'

            frame['compound_score'] = s['compound_score']
            frame['compound_score_sentiment'] = s['compound_score_sentiment']


            #frame.drop(['sentiment', 'sentiment_analyser'], axis = 1)
            frame.drop(frame.columns[[2, 3, 4]], axis = 1, inplace = True)
            frame.head()

            c=0
            for i in range(0,len(frame)):
                if frame['Cluster'][i]==neutral_n and frame['compound_score_sentiment'][i]=='Neutral':
                    c+=1
            ans_c=c/c0
            print(ans_c)

            d=0
            for i in range(0,len(frame)):
                if frame['Cluster'][i]==positive_n and frame['compound_score_sentiment'][i]=='Positive':
                    d+=1
            ans_d=d/d0
            print(ans_d)

            e=0
            for i in range(0,len(frame)):
                if frame['Cluster'][i]==negative_n and frame['compound_score_sentiment'][i]=='Negative':
                    e+=1

            ans_e=e/e0
            print(ans_e)

            ans=(ans_c*neu+ans_d*pos)/(c0+d0)
            print(ans)

            #data = [neu,pos,neg]
            data = {'Task' : 'Hours per Day', 'Positive' : pos, 'Negative' : neg, 'Neutral' : neu}

            return render_template('home.html', positive=pos, negative=neg, neutral=neu, data=data)
        # This part executes if server fails to response then display an Error Message
        except Exception as e:
            return '<h1>Bad Request : {}</h1>'.format(e)

    # if user make a simple GET request to this server it returns 'home.html' page
    else:
        return render_template('home.html',data=data)



# Running the server/app, when this file will be executed/run
if __name__ == "__main__":
    app.run(debug= True)
