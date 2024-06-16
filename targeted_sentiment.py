import argparse
import sys
import pandas as pd
import numpy as np

from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.tokenize import word_tokenize

def classify_sentiment(sentiment):
    ''' Classify sentiment as positive, negative, or neutral based on compound score. '''
    if sentiment >= 0.05:
        return 1 # positive
    elif sentiment <= -0.05:
        return -1 # negative
    else:
        return 0 # neutral


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
    

def get_weighted_sentiment(review, target_aspect):
    ''' Get sentiment for one review and one aspect, i.e. one cell of df. Give higher weights to words closer to the aspect.'''
    sid = SentimentIntensityAnalyzer()
    sentiment = 0
    rev_tokens = word_tokenize(review.lower())
    if target_aspect in rev_tokens:
        aspect_indices = [k for k, tok in enumerate(rev_tokens) if tok==target_aspect]
        for aspect_idx in aspect_indices:
            for i in range(len(rev_tokens)):
                if i == aspect_idx: continue  
                raw_sent_score = sid.polarity_scores(rev_tokens[i])['compound']
                if i > aspect_idx and any(negation in rev_tokens[aspect_idx:i] for negation in ['not', 'no', 'never']):
                    raw_sent_score *= -1
                weight = weight_term(aspect_idx - i, 'exp')
                sentiment += raw_sent_score * weight
    else: # Aspect not mentioned in review
        return None
    return sentiment

def aspect_sentiment_allreviews(reviews, target_aspect):
    ''' 
    Calculate the sentiment of the target_aspect for each review.

    @param reviews: list of reviews for one product
    @param target_aspect: the aspect to get sentiment on
    @return sentiment_across_revs: list of tuples (review, sentiment) for the target aspect
        ex. [(review 1, sentiment 1), (review 2, sentiment 2)]
    '''
    sentiment_across_revs = [] # List of tuples (review, sentiment) for the target aspect

    for r, review in enumerate(reviews):
        sentiment = get_weighted_sentiment(review, target_aspect)
        sentiment_across_revs.append((review, sentiment))

    return sentiment_across_revs

def analyze_reviews(reviews_df, target_aspects):
    ''' For each review, get the sentiment of each aspect. Return dataframe with rows as reviews and columns as aspects and asin.
    @param reviews: all reviews
    @param target_aspects: list of aspects to get sentiment on
    @return review_sentiments: list of list, aspect sentiments for each review, in order of target_aspects
    '''
    review_sentiments = []
    for i, row in reviews_df.iterrows():
        asin = row['parent_asin']
        review = row['text']
        # Get sentiment for each aspect (one row of df)
        aspect_sentiments = []
        for aspect in target_aspects:
            sent_curr_aspect = get_weighted_sentiment(review, aspect)
            aspect_sentiments.append(sent_curr_aspect)
        # review_sentiments.loc[i] = [asin, review, *aspect_sentiments]
        review_sentiments.append([asin, review, *aspect_sentiments])
        if i%5000==0: print(i, review_sentiments)
    return review_sentiments


def avg_targeted_sent(reviews, target_aspect):
    ''' 
    For each review, get the sentiment of the target_aspect. 
    Return the average sentiment across all reviews for the target_aspect.
    '''
    sentiments_across_revs = aspect_sentiment_allreviews(reviews, target_aspect)
    # Average sentiment across all reviews for the target_aspect
    sentiments_across_revs = [sent for rev, sent in sentiments_across_revs if sent is not None]
    if len(sentiments_across_revs) == 0: return None # No reviews of the product mention the aspect
    return sum(sentiments_across_revs) / len(sentiments_across_revs)


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
            sent_curr_aspect = avg_targeted_sent(product_reviews, aspect) # Product's sentiment for current aspect, across all reviews for the product
            aspect_sentiments.append(sent_curr_aspect) 
            if i%1000==0: print(f"Product: {product}, Aspect: {aspect}, Sentiment: {sent_curr_aspect}")
        product_to_sentiments[product] = aspect_sentiments 

    return product_to_sentiments


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Scrape product reviews from Amazon")
    parser.add_argument("review_file", help="csv file with product reviews")
    parser.add_argument("aspect_file", help="csv file with target aspects to get sentiment on")
    parser.add_argument("--product-level", help="get sentiment at product-level, not review-level", action="store_true")
    parser.add_argument("--output", help="csv file name to write the output to", default="targeted_sentiments.csv")
    args = parser.parse_args()
    review_file = args.review_file
    aspect_file = args.aspect_file

    # RUN Review-level: python targeted_sentiment.py handmade_reviews_balanced.csv handmade_noun_fts.csv --output review_targeted_sentiments.csv
    # RUN Product-level: python targeted_sentiment.py handmade_reviews_balanced.csv handmade_noun_fts.csv --product-level --output product_targeted_sentiments.csv

    reviews_df = pd.read_csv(review_file, header=0).dropna(subset=['text'])
    
    print("Reviews:\n", reviews_df.head())
    reviews_text = reviews_df['text']

    aspects = pd.read_csv(aspect_file)['noun'] # List of aspects to get sentiment on
    print("Aspects:\n", aspects)
    print("output:", args.output)
    if args.product_level:
        # Targeted sentiment at PRODUCT-level, averaged across reviews for the product
        product_sents = analyze_targeted_sentiment_products(reviews_df, aspects)
        sentiment_df = pd.DataFrame(product_sents, index=aspects.index).T
        sentiment_df.columns = aspects # Map column names to the aspect names 
        
    else:
        # Targeted sentiment at REVIEW-level, not averaged for a product
        review_sents = analyze_reviews(reviews_df, aspects) 
        sentiment_df = pd.DataFrame(review_sents, columns=['asin', 'review', *aspects])

    sentiment_df.to_csv(args.output)
    print("Review sentiments:\n", sentiment_df.head())
