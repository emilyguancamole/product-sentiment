# Writeup
# Introduction
This project sought to analyze the sentiment of product reviews on Amazon.

Lexicon-based

Targeted sentiment analysis was done to analyze the sentiment of a review towards a particular feature of the product. In other words, *what* about the product did the reviewer like or dislike?

# Methods 
Data used was the Amazon reviews dataset: https://amazon-reviews-2023.github.io/. This dataset contains products (identified by parent_asin), their star ratings (1-5 scale), reviews ('text' column), and other metadata. The data is separated into broad product categories, such as Cell_Phones_And_Accessories, Handmade, Books, etc. Each category has subcategories. The Handmade category was chosen for this project because the types of products it contains is very broad, spanning 646 subcategories. This increases the chance that the Handmade category represents products more generally, as opposed to a more niche category like Books.

The dataset was loaded using Huggingface in data_parser.py. Only products with more than 50 reviews were kept, with the assumption that products with a higher number of reviews will be more legitimate products and attract more legitimate reviewers. This resulted in 161979 total reviews in the base dataset. The base data was then saved as `handmade_reviews.csv`. EDA was done in eda.ipynb to analyze the characteristics of the dataset. This was saved as `handmade_reviews_processed.csv`.

Preprocessing of the reviews was performed in preprocess.py. Preprocessing steps were as follows. First, HTML tags were removed. Contractions were expanded using python's contractions library. Spacing after punctuations of `.` and `...` were adjusted to correct any spacing errors corrected. The reviews were converted to lowercase, then tokenized using NLTK's word tokenizer. Punctuation was removed. All words were lemmatized to their dictionary form using Spacy. Stopwords and numbers were removed. I experimented with extending the stopwords list to include words that are specific to the domain; i.e. words that would appear in every review. However, this extension would be more suitable for a more specific subcategory of products, such as including "case" and "phone" as stopwords for the subcategory Cell_Phones_And_Accessories->Basic_Cases. I found that the "Handmade" category overall was too broad to use domain-specific stopwords.

## Review-Level Sentiment Analysis
I performed review-level sentiment analysis, where each review was treated separately without grouping them into products. This was split into whole-review sentiment analysis and targeted sentiment analysis.

I noticed that the distribution of ratings was heavily skewed, with the majority of ratings being 5-stars. I balanced the dataset by rating such that there was a uniform distribution of ratings. This dataset was saved as `handmade_reviews_balanced.csv`. Note that this changed the distribution of the reviews per product; however, this was not important for review-level analysis.

    Balanced the reviews, since I saw that most predicted sentiments were > 0.1 (positive) and I wanted a more even distribution.
    - Note: this changed the distribution of reviews per product.

Whole-review sentiment analysis was performed in `whole_sentiment.py`. I used the Valence Aware Dictionary and Sentiment Reasoner (VADER) lexicon. This assigns the review a sentiment score between -1 (most negative) and +1 (most positive), calculated as the sum of all the lexicon ratings in the review. Evaluation of this model's performance was done in evaluate.ipynb; this discussion is below.

Targeted sentiment analysis was performed to analyze the sentiment of a review towards a particular feature of the product. To do this, feature extraction was first performed in `extract_features.py`. This was done by using Spacy to find all the nouns from all reviews along with their frequencies, creating a histogram of noun frequencies. Visuals of the histograms are in `eda.ipynb`. I selected nouns as features because they are most likely to be the aspects of a product that reviewers express sentiment about. Spacy extracted a total of 7235 nouns, 2069 of which occurred at least 5 times.
I also tried extracting noun phrases, i.e. sequencies of nouns that go together as one feature (e.g. "ankle bracelet"). However, the extractor struggled to find accurate noun phrases, and it often grouped together nouns that did not go together. Thus, noun phrase extraction is left as a task for future exploration.

I chose the features that occured with frequencies greater than 100 times. This was done to remove nouns that were likely "junk," with the assumption that nouns that were mentioned infrequently were quirks of individual reviews, rather than product features. This filtering resulted in 272 total features, which is about 13% of the nouns that occurred at least 5 times.
(The resulting list of features is found in `handmade_noun_fts.csv`).
Note that I will refer to "features" as "aspects" interchangeably in the rest of this report.
- removed "love" which was by far the most frequent "noun", though it should prob be a verb. 

Then, sentiment analysis was performed in `targeted_sentiment.py`. For each review, I calculated the sentiment of each aspect mentioned in the review by calculating a weighted sum of the sentiments of words around the target aspect. In this way, higher weights were given to words closer to the aspect. The VADER lexicon (wrapped in NLTK's SentimentIntensityAnalyzer) was used to calculate sentiment towards each aspect.
- FUTURE: try different weighting schemes... with a smaller dataset.

### Product-level targeted sentiment analysis
Product-level targeted sentiment analysis was done with the data `handmade_reviews_processed.csv` (before balancing the dataset, such that each product still had at least 50 reviews). I used the same set of product features. For each product, I evaluated the overall sentiment of each aspect, where the sentiment was averaged across all reviews for the product.
- CANT EVAULATE BC WOULD NEED AN AVERAGED, HANDLABELED ASPECT SENTIMENT SCORE AND THATS SOO LABOR INTENSIVE
COULD DO WHOLE-REVIEW SENTIMENT ANALYSIS BY PRODUCT INSTEAD
OR JUST NOT DO PRODUCT-LEVEL

# EVALUATION

## Whole review sentiment 
### Lexicon-based
Per review (not per product): comparison with rating.
Converted rating (1-5 scale) to a binary sentiment, where -1 is negative and +1 is positive sentiment. 

??Neutral was not used for whole review sentiments, since a rating of 3 is ambiguous and usually represents a more negative sentiment. 3-star ratings are usually interpreted as more negative [CITE], but __[do some analysis]__

Metrics---------------
When rating of 4,5 is positive; else negative (and predictions converted to binary)
TP:  10237
TN:  3302
FP:  12345
FN:  253
Precision:  0.45332565760340093
Recall:  0.9758817921830315
F1 score:  0.6190735365263668
Accuracy:  0.5180013008378926

When rating of 4,5 is positive; 3 is neutral; else negative (and predictions converted to pos/neu/neg)
TP:  9666
TN:  3201
FP:  9351
FN:  241
Precision:  0.5082820634169427
Recall:  0.9756737660240234
F1 score:  0.6683722859908726
Accuracy:  0.5729106371610491

When rating of 3,4,5 is positive; else negative
TP:  14874
TN:  2701
FP:  7793
FN:  868
Precision:  0.6561962324083469
Recall:  0.9448608817176979
F1 score:  0.7745059751620714
Accuracy:  0.6698810794328404

In all three cases, where was a high frequency of false positives, meaning that reviews were predicted as positive, when they were actually given negative ratings. This explains why increasing the range of ratings categorized as "positive" increased precision.

### Assumptions:
- Rating (number of stars) is an accurate "ground truth" label, i.e. consistent with the overall sentiment of the review. This may not always be true, EXAMPLE.


## Targeted sentiment
### Per review (not per product)

Got the features mentioned in the review and considered in the model. Hand-labeled each feature (before seeing the predicted label), trying to label with the actual sentiment expressed by the review towards the feature. 
Hand-labeling was trickier than anticipated. The actual meaning intended by the reviewer was often ambiguous, so I wasn't sure on some of the labels I gave. 
Some features detected by the model were not actually features; for instance, many of these words were detected as nouns but were not actually used in noun form in the context of the review. For these, I labeled the predicted sentiment as "na".

Metrics--------------
TP:  36
TN:  0
FP:  46
FN:  18
Precision:  0.43902439024390244
Recall:  0.6666666666666666
F1 score:  0.5294117647058824
Accuracy:  0.36

### Product-level targeted sentiment analysis
- Maybe can compare averaged #stars across all available reviews for that product with the 

### Limitations/Critiques
- Should have used a smaller dataset and experimented with more techniques, such as different weighting schemes.
- Should have refined my feature extraction process. Should have evaluated the resulting features before proceeding to use them to evaluate targeted sentiment.
- Evaluation of targeted sentiment was very manual and labor-intensive. Should streamline/automate this evaluation. 
-- This includes adding the unprocessed/original reviews to the dataframe of processed reviews, since I used the original reviews to manually label. (Because I didn't include them in the same dataframe, I had to search for the 20 test reviews manually.)


# Future Work
Use machine-learning.
Extend to other categories.
Improve feature selection