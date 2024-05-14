import pandas as pd
import contractions
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from bs4 import BeautifulSoup
import spacy
import re
from custom_definitions import CONTRACTIONS_MAP


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
    # Replace … with ...
    text = re.sub("…", "...", text)
    # if there's no space after /, add one
    text = re.sub(r'(?<=\w)/(?=\w)', '/ ', text)

    # Create a regular expression pattern to match contractions
    pattern = re.compile('({})'.format('|'.join(contraction_mapping.keys())), flags=re.IGNORECASE|re.DOTALL)
    
    # Use regular expression to substitute contractions with their expanded forms
    expanded_text = pattern.sub(expand_match, text)
    # Remove remaining apostrophes
    expanded_text = re.sub("'", "", expanded_text)

    return expanded_text

def lemmatize_review(review, lemmatizer_model='nltk', nlp=None):
    if lemmatizer_model == 'nltk':
        lemmatizer = WordNetLemmatizer()
        review = ' '.join([lemmatizer.lemmatize(word) for word in review])
    elif lemmatizer_model == 'spacy':
        doc = nlp(review)
        review = ' '.join([token.lemma_ for token in doc]) 
    
    return review

def remove_stopwords(review):
    stop_words = stopwords.words('english')
    # Add extra stopwords relevant to the domain
    # extra_stopwords = ['case', 'phone'] # TODO: add more, or make customizable
    # stop_words.extend(extra_stopwords)
    return ' '.join([word for word in review.split() if word.lower() not in stop_words])


def preprocess_review(review):
    # Remove HTML tags, like <br>
    review = BeautifulSoup(review, 'lxml').get_text()
    # review = expand_contractions(review, CONTRACTIONS_MAP)
    review = contractions.fix(review)
    
    # Correct spacing error after period punctuation
    review = re.sub(r'\.(?=[A-Z])', '. ', review)
    review = re.sub(r'\.\.\.', ' ', review) # Remove ellipses

    # Tokenize and lowercase #?? should i be lowercasing
    review = ' '.join([word.lower() for word in nltk.word_tokenize(review)])

    # Remove punctuation 
    review = re.sub(r'[^\w\s]', '', review) #except for !: re.sub(r'[^\w\s!]|(?<=\W)\s+', '', review) 
    
    # Lemmatize
    nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner']) # Load spacy model 'en'
    review = lemmatize_review(review, lemmatizer_model='spacy', nlp=nlp)

    # Remove stopwords
    review = remove_stopwords(review)

    # Remove numbers
    review = ' '.join([word for word in review.split() if not word.isdigit()])

    return review

def preprocess_all_reviews(reviews):
    '''
    Preprocess all reviews and change the reviews df
    @param reviews: df of reviews
    '''
    reviews.loc[:, 'text'] = reviews['text'].apply(preprocess_review)
    return reviews


def preprocess_and_save_in_batches(reviews_df, batch_size=10000, output_file='processed_reviews.csv'):
    total_reviews = len(reviews_df)
    processed_reviews = 0
    
    while processed_reviews < total_reviews:
        # Process reviews in batches
        batch_end = min(processed_reviews + batch_size, total_reviews)
        batch_df = reviews_df.iloc[processed_reviews:batch_end]
        
        # Preprocess the batch
        batch_df = preprocess_all_reviews(batch_df)
        
        # Save the batch to CSV
        mode = 'w' if processed_reviews == 0 else 'a'  # 'w' mode for the first batch, 'a' mode for subsequent batches
        header = True if processed_reviews == 0 else False  # Include header only for the first batch
        batch_df.to_csv(output_file, mode=mode, index=False, header=header)
        
        processed_reviews += batch_size
        print(f"Processed {processed_reviews}/{total_reviews}")

    
if __name__ == "__main__":
    # metadata = pd.read_csv('Basic_Cases_meta.csv')

    reviews_df = pd.read_csv('handmade_reviews.csv')
    print(f"Len handmade reviews: {len(reviews_df)}")
    # Remove review columns: images, user_id, timestamp, helpful_vote, verified_purchase
    reviews_df = reviews_df.drop(columns=['images', 'user_id', 'timestamp', 'helpful_vote', 'verified_purchase'])
    # Remove reviews that don't have text
    reviews_df = reviews_df.dropna(subset=['text'])

    # df_test = reviews_df[-5:] # Test on a small subset
    # print("Original reviews:")
    # print(df_test.head())
    # reviews_df = preprocess_all_reviews(df_test)

    # Preprocess all reviews
    preprocess_and_save_in_batches(reviews_df, batch_size=1000, output_file='handmade_reviews_processed.csv')

    


    