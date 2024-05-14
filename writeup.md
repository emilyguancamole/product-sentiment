# Writeup
## Introduction
This project sought to analyze the sentiment of product reviews on Amazon using lexicon-based methods. Whole-review analysis was performed to analyze the sentiment of the entire review. Additionally, targeted analysis was performed to evaluate the sentiment of a review towards a particular feature of the product. Both forms of sentiment analysis are important to businesses because they provide comprehensive insights into consumer perceptions and preferences, enabling businesses to understand the overall sentiment towards their products, as well as specific aspects that drive customer satisfaction or dissatisfaction.

## Methods 
Data used was the Amazon reviews dataset: https://amazon-reviews-2023.github.io/. This dataset contains products (identified by parent_asin), their star ratings (1-5 scale), reviews ('text' column), and other metadata. The data is separated into broad product categories, such as Cell_Phones_And_Accessories, Handmade, Books, etc. Each category has subcategories. The Handmade category was chosen for this project because the types of products it contains is very broad, spanning 646 subcategories. This increases the chance that the Handmade category represents products more generally, as opposed to a more niche category like Books.

The dataset was loaded using Huggingface in data_parser.py. Only products with more than 50 reviews were kept, with the assumption that products with a higher number of reviews will be more legitimate products and attract more legitimate reviewers. This resulted in 161979 total reviews in the base dataset. The base data was then saved as `handmade_reviews.csv`. EDA was done in eda.ipynb to analyze the characteristics of the dataset. This was saved as `handmade_reviews_processed.csv`.

Preprocessing of the reviews was performed in preprocess.py. Preprocessing steps were as follows. First, HTML tags were removed. Contractions were expanded using python's contractions library. Spacing after punctuations of `.` and `...` were adjusted to correct any spacing errors corrected. The reviews were converted to lowercase, then tokenized using NLTK's word tokenizer. Punctuation was removed. All words were lemmatized to their dictionary form using Spacy. Stopwords and numbers were removed. I experimented with extending the stopwords list to include words that are specific to the domain; i.e. words that would appear in every review. However, this extension would be more suitable for a more specific subcategory of products, such as including "case" and "phone" as stopwords for the subcategory Cell_Phones_And_Accessories->Basic_Cases. I found that the "Handmade" category overall was too broad to use domain-specific stopwords.

I noticed that the distribution of ratings was heavily skewed, with the majority of ratings being 5-stars. I balanced the dataset by rating such that there was a uniform distribution of ratings. This dataset was saved as `handmade_reviews_balanced.csv`. Note that this changed the distribution of number of reviews per product; however, this was not important for review-level analysis.

### Whole-Review Sentiment Analysis 
Whole-review sentiment analysis was performed in `whole_sentiment.py`. I used the Valence Aware Dictionary and Sentiment Reasoner (VADER) lexicon. This assigns the review a sentiment score between -1 (most negative) and +1 (most positive), calculated as the sum of all the lexicon ratings in the review. Evaluation of this model's performance was done in evaluate.ipynb; this discussion is below.

### Targeted Sentiment Analysis
Targeted sentiment analysis was performed to analyze the sentiment of a review towards a particular feature of the product. To do this, feature extraction was first performed in `extract_features.py`. This was done by using Spacy to find all the nouns from all reviews along with their frequencies, creating a histogram of noun frequencies. Visuals of the histograms are in `eda.ipynb`. I selected nouns as features because they are most likely to be the aspects of a product that reviewers express sentiment about. Spacy extracted a total of 7235 nouns, 2069 of which occurred at least 5 times.
I also tried extracting noun phrases, i.e. sequences of nouns that go together as one feature (e.g. "ankle bracelet"). However, the extractor struggled to find accurate noun phrases, and it often grouped together nouns that did not go together. Thus, noun phrase extraction is left as a task for future exploration.

I chose the features that occurred with frequencies greater than 100 times. This was done to remove nouns that were likely "junk," with the assumption that nouns that were mentioned infrequently were quirks of individual reviews, rather than product features. This filtering resulted in 272 total features, which is about 13% of the nouns that occurred at least 5 times.
(The resulting list of features is found in `handmade_noun_fts.csv`).
Note that I will refer to "features" as "aspects" interchangeably in the rest of this report.

Then, sentiment analysis was performed in `targeted_sentiment.py`. For each review, I calculated the sentiment of each aspect mentioned in the review by calculating a weighted sum of the sentiments of words around the target aspect. In this way, higher weights were given to words closer to the aspect. The VADER lexicon (wrapped in NLTK's SentimentIntensityAnalyzer) was used to calculate sentiment towards each aspect. Negations, such as "not" and "no," were taken into account by negating the resulting sentiment of the word.

<!-- ### Product-level targeted sentiment analysis
Product-level targeted sentiment analysis was done with the data `handmade_reviews_processed.csv` (before balancing the dataset, such that each product still had at least 50 reviews). I used the same set of product features. For each product, I evaluated the overall sentiment of each aspect, where the sentiment was averaged across all reviews for the product. -->
<!-- - CANT EVAULATE BC WOULD NEED AN AVERAGED, HANDLABELED ASPECT SENTIMENT SCORE AND THATS SOO LABOR INTENSIVE
COULD DO WHOLE-REVIEW SENTIMENT ANALYSIS BY PRODUCT INSTEAD
OR JUST NOT DO PRODUCT-LEVEL -->


## Evaluation and Discussion

### Whole-Review Sentiment Analysis
To evaluate the performance of the model on predicting the sentiment of whole reviews, I compared the predicted sentiment with the 5-star rating that accompanied the review. Because this treats the 5-star rating as a "ground truth" label, this assumes that the rating is consistent with the overall sentiment of the review.

Ratings (1-5 scale) were converted to a sentiment label. The classification of 3-star ratings was somewhat ambiguous. 3-star reviews are considered mixed, containing both positive and negative feedback. Therefore, I wanted to see how different classifications of ratings would affect the perceived performance of the model. I converted the rating to a binary label, where -1 is negative and +1 is positive sentiment, as well as to a ternary label, where I added 0 as neutral sentiment. Specific rules are listed below. Evaluation metrics included Precision, Recall, F1, and Accuracy.

#### Metrics
Metrics are presented in order of the most lenient definition of "positive" to most stringent.

When rating of 3,4,5 is positive; else negative
TP:  14874
TN:  2701
FP:  7793
FN:  868
Precision:  0.6561962324083469
Recall:  0.9448608817176979
F1 score:  0.7745059751620714
Accuracy:  0.6698810794328404

When rating of 4,5 is positive; 3 is neutral; else negative (and predictions converted to pos/neu/neg)
TP:  9666
TN:  3201
FP:  9351
FN:  241
Precision:  0.5082820634169427
Recall:  0.9756737660240234
F1 score:  0.6683722859908726
Accuracy:  0.5729106371610491

When rating of 4,5 is positive; else negative (and predictions converted to binary)
TP:  10237
TN:  3302
FP:  12345
FN:  253
Precision:  0.45332565760340093
Recall:  0.9758817921830315
F1 score:  0.6190735365263668
Accuracy:  0.5180013008378926

#### Discussion
In all three definitions of rating classes, there was a high frequency of false positives, meaning that reviews were predicted as having positive sentiment, when they were actually given negative ratings. This explains why a larger range of ratings categorized as "positive" increased precision and overall resulted in better quantitative metrics. The most significant change in these different definitions is the classification of a 3-star rating as positive, negative, or neutral. 
3-star ratings have a mixture of positive and negative feedback and are typically written in a more neutral tone [[source](https://stampede.ai/blog/why-3-star-reviews-actually-matter-more)]. They are usually interpreted as more negative by businesses and customers, as they deter sales [[source](https://fastercapital.com/content/Star-Rating--The-Impact-of-Star-Ratings-on-Consumer-Decision-Making.html)]. However, my model predicted a 3-star review as having "positive" sentiment 88% of the time. 
There are several possible reasons for this. Because reviewers of 3-star ratings frequently praise certain aspects of a product/service, positive sentiments are prevalent in 3-star reviews. The model may be more sensitive to these positive sentiments than the negative sentiments in the same review. Additionally, it is possible that reviewers of handmade products are more likely to use neutral or positive words, even when assigning a rating of 3.

[Nguyen et al.](https://scholar.smu.edu/cgi/viewcontent.cgi?article=1051&context=datasciencereview) also used the VADER lexicon for whole-review sentiment analysis of product reviews. They achieved a precision of 90% and recall of 89% in binary sentiment classification. It is interesting to note that they did not correct for skewed ratings, as their dataset also had an overwhelmingly high proportion of 5-star ratings (and thus reviews classified as "positive" (+1)). Because my model showed good ability to detect "positive" reviews (recall of 94.5-97.6%), it may perform better when evaluated on a skewed dataset.

#### Assumptions:
As noted above, during evaluation, I compared the predicted sentiment with the 5-star rating that accompanied the review. This treats the 5-star rating as a "ground truth" label and therefore assumes that the rating is consistent with the overall sentiment of the review. (This may not always be true. For instance, the review "This was a gift." has a very neutral sentiment, but the reviewer gave it a rating of 5.)


### Targeted Sentiment Analysis
Evaluation of performance in targeted sentiment analysis was done with manual labeling of each feature. A "test set" was constructed by randomly sampling 20 reviews, with 4 reviews from each star rating. I first retrieved the features that were both mentioned in the review and considered in the model (i.e. included in `handmade_noun_fts.csv`). I used the original reviews (before preprocessing) to hand-label each feature (before seeing the predicted label) as positive (+1), negative (-1), or neutral (0). When hand-labeling, I tried to label the feature with the actual sentiment expressed by the review towards the feature. 
Some features detected by the model were not actually features; for instance, some words were detected as nouns but were not actually used in noun form in the context of the review. For these, I labeled the predicted sentiment as "na".

I converted the predicted sentiment score for each aspect to ternary sentiment classes (positive, negative, or neutral). Ternary labeling was chosen because this is a "middle ground" in terms of strictness of a "positive" sentiment class. For all aspects that were truly features of the product (i.e. not "na"), I calculated precision, recall, F1, and accuracy of the model's predictions.

#### Metrics
TP:  36
TN:  0
FP:  46
FN:  18
Precision:  0.43902439024390244
Recall:  0.6666666666666666
F1 score:  0.5294117647058824
Accuracy:  0.36

### Discussion
Overall, targeted sentiment did not perform as well as whole-review sentiment analysis. Similarly to the whole-review sentiment analysis, there was a high rate of false positives, suggesting that the model was more likely to predict the review as having a positive sentiment toward any aspect.

Hand-labeling the sentiments of product features was trickier than anticipated. The actual meaning intended by the reviewer was often ambiguous, so I was not confident some of the labels I assigned to features. Thus, the "ground truth" labels produced by hand-labeling may not reflect the true sentiment intended by the reviewer.

Additionally, the test set used only contained 20 reviews, which is a very small sample compared to the 161000+ total reviews. A small test set was used because hand-labeling the reviews was a time-consuming and manual process. A different test set may result in different metrics, and a larger test set may result in more reliable metrics that reflect the true performance of the model.


## Conclusion
Overall, my lexicon-based model was able to detect reviews with positive sentiment fairly well. However, it struggled with a high false positive rate, indicating that more measures for detecting negative sentiment should be incorporated. Detection of whole-review sentiment performed significantly better than targeted sentiment. The approach used for lexicon-based targeted sentiment analysis should be explored further, as most existing literature I found focused on whole-review analysis, but targeted sentiment may be more useful for businesses.
From this project's evaluation, it is apparent that the method for evaluation should be chosen carefully (and before the actual evaluation is performed), since this significantly affects the quantitative performance of the model.

### Limitations/Critiques
One limitation of this project is the quality of the product features extracted. A simple frequency- and POS-based method was employed. However, some of these features were not actual aspects that the reviewer expressed sentiment on. Additionally, the dataset category used in this project, Handmade products, contained reviews of products that spanned a wide range of subcategories. Although I chose this for the breadth, this potentially made it more difficult to select high-quality features, since there was likely a greater variety of features. In the future, extracting features from specific subcategories and applying them to subcategory-specific sentiment analysis may result in better performance. Using more sophisticated feature extraction methods, such as finding noun phrases or using context-aware NLP algorithms, can also be explored to obtain better product features.

Additionally, the dataset I used was very large, with over 161000 reviews even after balancing the ratings. This long preprocessing and analysis times (hours), limiting the flexibility of the model. One change I would employ is using a smaller dataset, then experimenting with a wider variety of techniques, such as different weighting schemes in targeted sentiment analysis. 

Finally, evaluation of the targeted sentiment task was very manual and labor-intensive. this process should be streamlined and automated. This includes adding the unprocessed/original reviews to the dataframe of processed reviews, since I used the original reviews to manually label. (Because I didn't include them in the same dataframe, I had to search for the 20 test reviews manually.)


### Future Work
There are vast possibilities for future extensions to this project besides the ones discussed above. 

Much previous literature explores the use of machine learning in sentiment analysis, where a ML model is trained using examples of human language (e.g. product reviews) and their associated sentiment. [Nguyen et al.](https://scholar.smu.edu/cgi/viewcontent.cgi?article=1051&context=datasciencereview) compare lexicon-based methods to logistic regression, SVM, and gradient boosting models, and they found that ML methods performed better. The disadvantage of ML-based methods is its need for large amounts of training data; however, this is not a problem when working with product reviews. It would also be interesting to evaluate more complex ML models, such as transformers, which can handle long-range dependencies and may more accurately capture sentiment in complicated reviews.

Adding other categories of products and extracting high-quality features from their reviews can improve the generalizability of the model as well. This may address the problem in sentiment classification where models trained in one domain often struggle in another domain [[source](https://american-cse.org/csci2015/data/9795a175.pdf)]. 

In this project, only words were included in processed reviews; punctuation, numbers, and other miscellaneous characters were removed. Future extensions can investigate the inclusion of these features. In particular, punctuation, such as exlamation points, carry sentiment and can be included. Additionally, emojis were frequently used in the product reviews and carry sentiment. Many NLP libraries are able to analyze emojis, so these can be employed.


## References
- https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023
- https://stampede.ai/blog/why-3-star-reviews-actually-matter-more
- https://fastercapital.com/content/Star-Rating--The-Impact-of-Star-Ratings-on-Consumer-Decision-Making.html
- https://scholar.smu.edu/cgi/viewcontent.cgi?article=1051&context=datasciencereview
- https://american-cse.org/csci2015/data/9795a175.pdf
- https://nycdatascience.com/blog/student-works/learning-category-wise-product-features-from-amazon-reviews/
