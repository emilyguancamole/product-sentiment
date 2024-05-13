import sys
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

# SentimentIntensityAnalyzer 

def collect_data(urls):
    ''' Take in list of urls (product review pages) to search for, 
    then use data_collector to get the product info, which saves to a csv '''
    for url in urls:
        data_collector.get_product_info(url)

    
def get_whole_sentiment(reviews):
    ''' Get the sentiment of the entire review '''
    sia = SentimentIntensityAnalyzer()
    sentiment = []
    for i in range(len(reviews)):
        sentiment.append(sia.polarity_scores(reviews[i]))
    return sentiment

def weight_term(distance, weight_type):
    ''' Compute the weight of a term based on its distance from the target word. '''
    distance = abs(distance)
    if weight_type == 'uniform': # part 1 of hw uses this
        return 1
    elif weight_type == 'exp':
        return 1 / (np.exp(distance))
    elif weight_type == 'stepped':
        if distance == 1:
            return 6.0
        elif distance==2 or distance==3:
            return 3.0
        else:
            return 1.0
    else: # weighting scheme of my own choice
        return 1 / (distance + 1) # linearly decreasing weight as distance increases
    
def aspect_sentiment(reviews, target_aspect):
    ''' 
    Calculate the sentiment of the target_aspect, using words around it, for each review.
    Give higher weights to words closer to the aspect.

    @param reviews: list of reviews for one product
    @param target_aspect: the aspect to get sentiment on
    @return sentiment_across_revs: list of tuples (review, sentiment) for the target aspect
        ex. [(review 1, sentiment 1), (review 2, sentiment 2)]
    '''
    sid = SentimentIntensityAnalyzer() 
    sentiment_across_revs = [] # List of tuples (review, sentiment) for the target aspect

    for r, review in enumerate(reviews):
        sentiments_of_review = []
        # if r % 1000==0: print("Review:", review)
        tokens = word_tokenize(review.lower()) #? should already be lowercase
        # If the aspect is mentioned, get its sentiment in the review
        if target_aspect.lower() in tokens: 
            # Indices of occurences of the target aspect
            aspect_indices = [k for k, tok in enumerate(tokens) if tok==target_aspect.lower()]
            for aspect_idx in aspect_indices:
                # Get sentiment of words around the aspect
                for i in range(len(tokens)): #range(max(0, aspect_idx - 3), min(len(tokens), aspect_idx + 4)):
                    if i == aspect_idx: continue
                    raw_sent_score = sid.polarity_scores(tokens[i])['compound']
                    # Account for negations
                    if i > aspect_idx and any(negation in tokens[aspect_idx:i] for negation in ['not', 'no', 'never']):
                        raw_sent_score *= -1
                        
                    # Weight based on distance from target
                    weight = weight_term(aspect_idx - i, 'exp')
                    
                    sentiments_of_review.append(raw_sent_score * weight)

            # Overall sentiment score for the aspect in the current review
            if r % 1000==0: print("Review:", review, "Sentiments:", sentiments_of_review)
            overall_sentiment_rev = sum(sentiments_of_review)
            sentiment_across_revs.append((review, overall_sentiment_rev))
        else:
            sentiment_across_revs.append((review, None))

    if sentiment_across_revs:
        # Averge sentiment for the aspect
        aspect_avg_sent = np.mean([sent for rev, sent in sentiment_across_revs if sent is not None]) 
    else:
        aspect_avg_sent = None # No reviews of the product mention the aspect
    return aspect_avg_sent
    

def analyze_targeted_sentiment_products(reviews, target_aspects):
    ''' For each product, get the sentiment of each aspect averaged across all reviews for the product.'''
    product_to_sentiments = {} # Dict of product to its list of sentiments for each aspect. Sentiments are listed in order of target_aspects
    
    unqiue_products = reviews['parent_asin'].unique()
    for i, product in enumerate(unqiue_products):
        # if i % 1000==0: print("Product:", product)
        product_reviews = reviews[reviews['parent_asin'] == product]['text']
        # Each product has a list of targeted sentiments for each aspect that exists
        aspect_sentiments = []
        for aspect in target_aspects:
            sent_curr_aspect = aspect_sentiment(product_reviews, aspect) # Product's sentiment for current aspect, across all reviews for the product
            if i%1000==0: print(f"Product: {product}, Aspect: {aspect}, Sentiment: {sent_curr_aspect}")
            aspect_sentiments.append(sent_curr_aspect) 
        product_to_sentiments[product] = aspect_sentiments 

    return product_to_sentiments

def analyze_reviews(reviews_df, target_aspects):
    ''' For each review, get the sentiment of each aspect. Return dataframe with rows as reviews and columns as aspects and asin.
    @param reviews: all reviews
    @param target_aspects: list of aspects to get sentiment on
    '''
    review_sentiments = pd.DataFrame(columns=['asin', 'review', *target_aspects])
    for i, row in reviews_df.iterrows():
        asin = row['parent_asin']
        review = row['text']
        # if i % 1000==0: print("Review:", review)
        aspect_sentiments = []
        for aspect in target_aspects:
            sent_curr_aspect = aspect_sentiment([review], aspect)
            aspect_sentiments.append(sent_curr_aspect)
        review_sentiments.loc[i] = [asin, review, *aspect_sentiments]
        if i % 1000==0: print(f"Review: {review}, Sentiments: {aspect_sentiments}")
    return review_sentiments



if __name__ == "__main__":
    review_file = sys.argv[1]
    aspect_file = sys.argv[2]
    #RUN: python sentiment.py handmade_reviews_balanced.csv handmade_noun_fts.csv

    reviews_df = pd.read_csv(review_file, header=0).dropna(subset=['text'])
    
    print("Reviews:\n", reviews_df.head())
    reviews_text = reviews_df['text']

    aspects = pd.read_csv(aspect_file)['noun'] # List of aspects to get sentiment on
    print("Aspects:\n", aspects)

    # ex_aspect = [aspects.index[0]] # Example
    # print("Example aspect:", ex_aspect)

    # Targeted sentiment at product-level, averaged across reviews for the product
    # product_sents = analyze_targeted_sentiment_products(reviews_df, aspects)
    # sentiment_df = pd.DataFrame(product_sents, index=aspects.index).T
    # sentiment_df.columns = aspects # Map column names to the aspect names
    # sentiment_df.to_csv("product_targeted_sentiments.csv")
    
    # Targeted sentiment at review-level, not averaged for a product
    review_sents = analyze_reviews(reviews_df, aspects)
    sentiment_df_unique = pd.DataFrame(review_sents, index=aspects.index).T
    sentiment_df_unique.columns = aspects
    sentiment_df_unique.to_csv("product_targeted_sentiments_unique.csv")
