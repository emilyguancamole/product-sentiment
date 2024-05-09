from collections import Counter
import pandas as pd
import spacy
import nltk
        

# POS tagger with spacy #* i think this is better, but still has issues. how to make better?

def extract_nouns_spacy(text):
    ''' Extract nouns from text using spacy, save to csv. '''
    nlp = spacy.load('en_core_web_sm')
    # pull out all noun, make histogram of occurrences
    nouns = []
    for review in reviews_df['text']:
        doc = nlp(review)
        for token in doc:
            if token.pos_ == 'NOUN':
                nouns.append(token.text)
#     noun_phrases = []
#     for chunk in doc.noun_chunks:
#         noun_phrases.append(chunk.text)

    # Save as csv
    nouns_series = pd.Series(nouns)
    nouns_series.value_counts().to_csv('nouns_spacy.csv') 
    return nouns

def top_k_nouns(nouns, k):
    noun_counts = Counter(nouns)
    top_k_features = noun_counts.most_common(k)
    print("Top", k, "potential product features:")
    for feature, count in top_k_features:
        print(f"{feature}: {count} occurrences")

    return top_k_features # List of tuples (feature, count)


# POS tag each review NLTK

# tagged_reviews = [nltk.pos_tag(nltk.word_tokenize(review)) for review in reviews_df['text']]
# def extract_noun_sequences(tagged_words):
#     noun_sequences = []
#     current_sequence = []
#     for word, pos_tag in tagged_words:
#         if pos_tag.startswith('NN'):
#             current_sequence.append(word)
#         elif current_sequence:
#             noun_sequences.append(' '.join(current_sequence))
#             current_sequence = []
#     if current_sequence:  # Handle the case where the last word(s) form a noun sequence
#         noun_sequences.append(' '.join(current_sequence))
#     return noun_sequences
# all_noun_sequences = [sequence for tagged_review in tagged_reviews for sequence in extract_noun_sequences(tagged_review)]
# noun_sequence_counts = Counter(all_noun_sequences)
# # sort by descending count
# noun_sequence_counts = pd.Series(noun_sequence_counts).sort_values(ascending=False)
# noun_sequence_counts.to_csv('noun_sequences_nltk.csv')


#** NOTE: goal is to create a list of features for products in general

if __name__ == "__main__":
    reviews_df = pd.read_csv('Basic_Cases_reviews_processed.csv', index_col=0)
    # remove reviews that don't have text
    reviews_df = reviews_df.dropna(subset=['text'])

    # Extract nouns from reviews, get top k features
    nouns_cases = extract_nouns_spacy(reviews_df['text'])
    top_k_nouns(nouns_cases, 10)