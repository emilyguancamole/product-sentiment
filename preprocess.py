import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from bs4 import BeautifulSoup
import re
from custom_definitions import CONTRACTIONS_MAP

from collections import Counter


def expand_contractions(text, contraction_mapping):
    """
    Expand contractions in a text, using a mapping of contractions to their expanded forms.
    """
    def expand_match(contraction):
        """
        Expand one contraction match using a statically-defined mapping for contractions to expansions.
        @param contraction (str): Contraction to expand.
        @return str: Expanded form of the contraction.
        """
        match = contraction.group(0)
        first_char = match[0]
        expanded_contraction = contraction_mapping.get(match) if contraction_mapping.get(match) else contraction_mapping.get(match.lower())
        expanded_contraction = first_char + expanded_contraction[1:] # Keep the first character of the contraction
        return expanded_contraction
    
    # Replace unicode character ’ (found in quite a few reviews) with '
    text = re.sub("’", "'", text)

    # Create a regular expression pattern to match contractions
    pattern = re.compile('({})'.format('|'.join(contraction_mapping.keys())), flags=re.IGNORECASE|re.DOTALL)
    
    # Use regular expression to substitute contractions with their expanded forms
    expanded_text = pattern.sub(expand_match, text)
    # Remove remaining apostrophes
    expanded_text = re.sub("'", "", expanded_text)

    return expanded_text

def remove_stopwords(review):
    stop_words = stopwords.words('english')
    # Add extra stopwords relevant to the domain
    extra_stopwords = ['case', 'phone'] # TODO: add more, or make customizable
    stop_words.extend(extra_stopwords)
    return ' '.join([word for word in review.split() if word.lower() not in stop_words])

def preprocess_review(review):
    # Remove HTML tags, like <br>
    review = BeautifulSoup(review, "html.parser").get_text() 

    # cont = Contractions(api_key="glove-twitter-100") #todo: more elaborate contraction, but had trouble installing (java version issue?)
    # cont.load_models()
    # review = list(cont.expand_texts([review]))[0]
    review = expand_contractions(review, CONTRACTIONS_MAP)
    
    # Correct spacing error after period punctuation ?? don't need bc we tokenize
    # review = re.sub(r'\.(?=[A-Z])', '. ', review)
    # review = re.sub(r'\.\.\.', ' ', review) # Remove ellipses

    print("Review after contraction expansion:", review)

    #? Tokenize
    review = nltk.word_tokenize(review)

    # Lemmatize
    lemmatizer = WordNetLemmatizer()
    review = ' '.join([lemmatizer.lemmatize(word) for word in review])
    print("Review after lemmatization:", review)
    # Stemming? not sure if this is necessary / makes sense

    # Remove stopwords
    review = remove_stopwords(review)
    print("Review after stopword removal:", review)
    # Remove numbers
    review = ' '.join([word for word in review.split() if not word.isdigit()])
    print("Review after number removal:", review)

    return review

def preprocess_all_reviews(reviews):
    '''
    Preprocess all reviews and change the reviews df
    @param reviews: df of reviews
    '''
    reviews.loc[:, 'text'] = reviews['text'].apply(preprocess_review)
    return reviews


# TODO 5/5 POS tagger, pull out all noun sequences, make histogram of occurrences, take top k (remove junk)

def get_nouns(text):
    nouns = []
    for sentence in nltk.sent_tokenize(text): #? Tokenize into sentences, then words
        for word, pos in nltk.pos_tag(nltk.word_tokenize(sentence)):
            if pos.startswith('NN'):  # NN is nouns
                nouns.append(word)
    return nouns

def get_nouns_all_reviews(reviews):
    all_nouns = []
    for i, review in enumerate(reviews):
        if i%10==0: print(i)
        all_nouns += get_nouns(review)
    return all_nouns


if __name__ == "__main__":
    # metadata = pd.read_csv('Basic_Cases_meta.csv')
    reviews_df = pd.read_csv('Basic_Cases_reviews.csv')

    # Remove review columns: images, user_id, timestamp, helpful_vote, verified_purchase
    reviews_df = reviews_df.drop(columns=['images', 'user_id', 'timestamp', 'helpful_vote', 'verified_purchase'])
    # Remove reviews that don't have text
    reviews_df = reviews_df.dropna(subset=['text'])

    df_test = reviews_df[-5:]
    print("Original reviews:")
    print(df_test.head())
    reviews_df = preprocess_all_reviews(df_test)

    # Preprocess reviews
    # reviews_df = preprocess_all_reviews(reviews_df)
    print("Preprocessed reviews:")
    print(reviews_df.head())

    # save to csv
    reviews_df.to_csv('test.csv')
    # reviews_df.to_csv('Basic_Cases_reviews_processed.csv')


    