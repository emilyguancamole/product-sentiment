import pandas as pd
import numpy as np
import nltk

import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import string

import data_collector

#SentimentIntensityAnalyzer to get the neg/neu/pos scores of the text.
# This uses a "bag of words" approach:
# Stop words are removed
# each word is scored and combined to a total score.

def collect_data(urls):
    ''' Take in list of urls (product review pages) to search for, 
    then use data_collector to get the product info, which saves to a csv '''
    for url in urls:
        data_collector.get_product_info(url)


def process_data(file):
    ''' Take in a csv file of product info, and process reviews by tokenizing, removing stopwords, stemming '''
    df = pd.read_csv(file)
    reviews = df["review"]
    sw = set(stopwords.words('english')) 
    processed_reviews = []
    for review in reviews:
        tokens = word_tokenize(review)
        tokens = [word.lower() for word in tokens]
        # Remove stopwords and punctuation
        tokens = [token for token in tokens if token not in sw and token not in string.punctuation]
        #?? Stemming - does this make sense to do
        # stemmer = PorterStemmer()
        # tokens = [stemmer.stem(token) for token in tokens]

        processed_reviews.append(' '.join(tokens))
        
    return processed_reviews
    
def get_whole_sentiment(reviews):
    ''' Get the sentiment of the entire review '''
    sia = SentimentIntensityAnalyzer()
    sentiment = []
    for i in range(len(reviews)):
        sentiment.append(sia.polarity_scores(reviews[i]))
    return sentiment    


# Aspect-based:
# ner and pos tagging to extract relevant entities
# Analyze sentiment of each identified aspect separtely
# Consider weighting sentiment scores based on the importance or relevance of each aspect

def analyze_targeted_sentiment(reviews, target_aspect):
    # TODO stem target_aspect
    
    sid = SentimentIntensityAnalyzer()
    aspect_sentiments = []
    for review in reviews:
        sentences_in_review = sent_tokenize(review)
        aspect_sent = []
        for sent in sentences_in_review:
            sentiment_score = None
            if target_aspect.lower() in sent.lower():
                sentiment_score = sid.polarity_scores(sent)['compound']
                aspect_sent.append(sentiment_score)
        #?? add aspect's mean? sentiment score for the whole review
        if sentiment_score: # only add if the aspect was found in the review
            aspect_sentiments.append((review, np.mean(aspect_sent)))
    return aspect_sentiments

# Example usage:
product_review = "The battery life of this laptop is excellent, but the screen quality could be better." # TODO: tokenize by groups of words? bc "could be better" is being interpreted as positive
target_aspect = "screen"


if __name__ == "__main__":
    reviews = process_data("product_reviews.csv")
    # sentiment = get_whole_sentiment(reviews)
    # print(sentiment)
    aspect_sentiments = analyze_targeted_sentiment(reviews, "cost")
    for review, sentiment_score in aspect_sentiments:
        print(f"Review: {review[:100]}")
        print(f"Sentiment Score: {sentiment_score}")
        print("----")

