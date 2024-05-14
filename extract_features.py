from collections import Counter
import pandas as pd
import spacy
import nltk
        

# POS tagger with spacy #* i think this is better, but still has issues. how to make better?

def extract_nouns_spacy(text, filename_write):
    ''' Extract nouns from text using spacy, save to csv. 
    @text: list of text to extract nouns from'''
    nlp = spacy.load('en_core_web_sm')
    # Pull out all nouns
    nouns = []
    for review in text:
        doc = nlp(review)
        for token in doc:
            if token.pos_ == 'NOUN':
                nouns.append(token.text)
#     noun_phrases = []
#     for chunk in doc.noun_chunks:
#         noun_phrases.append(chunk.text)

    # Save as csv
    nouns_series = pd.Series(nouns)
    nouns_series.value_counts().to_csv(filename_write) 
    return nouns

def top_k_nouns(nouns, k, filename_write=None):
    noun_counts = Counter(nouns)
    top_k_features = noun_counts.most_common(k)
    print("Top", k, "potential product features:")
    for feature, count in top_k_features:
        print(f"{feature}: {count} occurrences")

    if filename_write:
        pd.Series(dict(top_k_features)).to_csv(filename_write)

    return top_k_features # List of tuples (feature, count)

def filter_nouns_freq(df, k=100, filename_write=None):
    ''' Filter out nouns that occur less than k times '''
    # Filter df to only have nouns with > k occurrences
    df_filt = df[df['count'] > k]
    print(f"Length of features with freq > {k}: {len(df_filt)}")
    df_filt = df_filt.rename(columns={'Unnamed: 0': 'noun'})
    if filename_write:
        df_filt.to_csv(filename_write, index=False)
    return df_filt

def extract_noun_chunks(text, filename_write):
    ''' Extract noun chunks from text using spacy. Only keep the ones that occur >1 times. Save to csv. 
    NOTE!!!!! this didnt work, likely bc preprocessed to take out stuff beforehand.
        ex. "mother day gift discern daughterinlaw thrill" was extracted as ONE noun chunk
    @text: list of text to extract noun chunks from'''
    nlp = spacy.load('en_core_web_sm')
    # Pull out all noun chunks
    noun_chunks = set()
    for review in text:
        doc = nlp(review)
        for chunk in doc.noun_chunks:
            noun_chunks.add(chunk.text)
            print(chunk.text)

    # Only keep noun chunks that occur >1 times
    noun_chunk_counts = Counter(noun_chunks)
    noun_chunks = [chunk for chunk, count in noun_chunk_counts.items() if count > 1]
    noun_chunks_series = pd.Series(noun_chunks)
    noun_chunks_series.value_counts().to_csv(filename_write) 
    return noun_chunks




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
    # reviews_df = pd.read_csv('Basic_Cases_reviews_processed.csv', index_col=0)
    reviews_df = pd.read_csv('handmade_reviews_processed.csv').dropna(subset=['text'])

    # Extract nouns from reviews, get top k features
    print("NOUNS--------------------")
    # nouns_extracted = extract_nouns_spacy(reviews_df['text'], 'handmade_nouns_spacy.csv')
    # top_k_nouns(nouns_extracted, 10)
    # print("\nCHUNKS--------------------")
    # noun_chunks = extract_noun_chunks(reviews_df['text'], 'handmade_noun_chunks_spacy.csv')
    # top_k_nouns(noun_chunks, 10)

    nouns_extracted = pd.read_csv('handmade_nouns_bal.csv')
    filter_nouns_freq(nouns_extracted, 100, 'handmade_noun_fts.csv')