import sys
import pandas as pd
import numpy as np

import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.tokenize import word_tokenize
import data_collector

def collect_data(urls):
    ''' Take in list of urls (product review pages) to search for, 
    then use data_collector to get the product info, which saves to a csv '''
    for url in urls:
        data_collector.get_product_info(url)

    
def get_whole_sentiment(reviews):
    ''' For all reviews, get sentiment of the entire review. 
    @param reviews: list of reviews
    @return sentiment_per_review: dataframe with each review and its overall sentiment '''
    sia = SentimentIntensityAnalyzer()
    sentiment_per_review = pd.DataFrame(columns=['review', 'sentiment'])
    for i, review in enumerate(reviews):
        sentiment = sia.polarity_scores(review)['compound']
        sentiment_per_review.loc[i] = [review, sentiment]
    return sentiment_per_review

def classify_sentiment_binary(sentiment):
    ''' Classify sentiment as positive or negative based on compound score. '''
    if sentiment >= 0:
        return 1
    else:
        return -1
    

if __name__ == "__main__":
    # RUN: python whole_sentiment.py handmade_reviews_balanced.csv handmade_noun_fts.csv
    review_file = sys.argv[1]
    aspect_file = sys.argv[2]
    review_df = pd.read_csv(review_file).dropna(subset=['text'])
    review_df.head()
    sentiments_df = get_whole_sentiment(review_df['text']) # df

    print(sentiments_df)
    sentiments_df.to_csv('whole_sentiment_output.csv', index=False)




