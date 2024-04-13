import pandas as pd
import numpy as np
import nltk

from nltk.sentiment import SentimentIntensityAnalyzer
from tqdm.notebook import tqdm

#SentimentIntensityAnalyzer to get the neg/neu/pos scores of the text.
# This uses a "bag of words" approach:
# Stop words are removed
# each word is scored and combined to a total score.

df = pd.read_csv('../input/amazon-fine-food-reviews/Reviews.csv')
df = df.head(500)
example = df['Text'][50]
print(example)

def process_data(file):
    df = pd.read_csv(file)
    #