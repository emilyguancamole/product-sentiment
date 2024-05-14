## Writeup

Amazon reviews dataset.

Got only products with > 50 reviews, just as a filtering mechansim. assumption is that products with a higher number of reviews will be more legitimate products and attract more legitimate reviewers.
Preprocess reviews to remove stopwords etc. 

Balanced the reviews, since I saw that most predicted sentiments were > 0.1 (positive) and I wanted a more even distribution.
- Note: this changed the distribution of reviews per product.


feature extraction. removed "love" which was by far the most frequent "noun", though it should prob be a verb.
- Chose the features with >100 freq. Justification: 7235 total nouns - 5166 with freq<5 = 2069. 272 is about 13%.


# EVALUATION
## Targeted sentiment
Per review (not per product) - todo
- manual comparison

Per product - done
- not really evaluation, just looking. 
- Maybe can compare averaged #stars across all available reviews for that product with the 

## Whole review sentiment 
### Lexicon-based
Per review (not per product): comparison with rating.
Converted rating (1-5 scale) to a binary sentiment, where -1 is negative and +1 is positive sentiment. 

??Neutral was not used for whole review sentiments, since a rating of 3 is ambiguous and usually represents a more negative sentiment. 3-star ratings are usually interpreted as more negative [CITE], but __[do some analysis]__


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

Assumptions:
- Rating (number of stars) is an accurate "ground truth" label, i.e. consistent with the overall sentiment of the review. This may not always be true, EXAMPLE.
