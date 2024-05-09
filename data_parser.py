from datasets import load_dataset
import pandas as pd

''' Get metadata and review data for a specified category, save to csv.

'''
def get_category_reviews(metadata, review_data, category):
    ''' Get the metadata and review data corresponding to the specified category 
    @params metadata, review_data: class 'datasets.arrow_dataset.Dataset'
    '''

    category_name = category.replace('&','and').replace(' ','_')
    print("Category name", category_name)

    # Get meta data for the target category - go though ALL products in the category
    # meta_subset = []
    # for rev in metadata:
    #     if category in rev['categories']:
    #         meta_subset.append(rev)
    # meta_df = pd.DataFrame(meta_subset)
    # print("meta head:\n", meta_df.head())
    # meta_df.to_csv(category_name + '_meta.csv')

    # Read in review data for products (asin) in category - only first 10000 reviews for speed
    meta_df = pd.read_csv(category_name + '_meta.csv')
    relevant_asins = meta_df['parent_asin'].tolist()
    print("Relevant asins", len(relevant_asins), "\n", relevant_asins[:15])

    relevant_revs = []
    for i, rev in enumerate(review_data):
        if rev['parent_asin'] in relevant_asins:
            relevant_revs.append(rev)
        if i%1000==0: print(i)
        if i>10000: break # Only get first 10000 reviews
    print("relevant_revs", relevant_revs[0])
    
    relevant_revs_df = pd.DataFrame(relevant_revs)
    relevant_revs_df.to_csv(category_name + '_reviews.csv', index=False)
    return meta_df, relevant_revs_df

def get_multiple_reviews_csv(metadata, review_data, category_name):
    ''' 
    Get review data with more than 5 reviews per product, for all subcategories in the overall category.
    @params metadata, review_data: class 'datasets.arrow_dataset.Dataset'
    '''
    meta_df = pd.DataFrame(metadata)
    print("meta head:\n", meta_df.head())
    reviews_df = pd.DataFrame(review_data)
    print(f"{category_name} reviews:\n", reviews_df.head())
    print("Reviews shape", reviews_df.shape)

    # Only keep products (parent_asin) with more than 5 reviews
    review_counts = reviews_df['parent_asin'].value_counts()
    print("Review counts", review_counts)
    relevant_asins = review_counts[review_counts>5].index.tolist()
    print(f"{len(relevant_asins)} relevant asins")
    
    relevant_reviews_df = reviews_df[reviews_df['parent_asin'].isin(relevant_asins)]
    if relevant_reviews_df.empty:
        print("No products with more than 5 reviews found.")
        return
    print("Relevant_reviews_df shape", relevant_reviews_df.shape)
    relevant_reviews_df.to_csv(category_name + '_reviews.csv', index=False)
    print("Filtered review data saved as", category_name + '_reviews.csv')


if __name__ == "__main__":
    '''
    Read in raw review data and metadata for a specified category, save to csv.
    Data: Amazon product review: online dataset
    '''

    # Load "full" splits of data and metadata
    
    # CELLPHONE DATA, SUB CATEGORY: Basic Cases ------------------
    # phone_review_data = load_dataset("McAuley-Lab/Amazon-Reviews-2023", "raw_review_Cell_Phones_and_Accessories", split="full", trust_remote_code=True)
    # phone_review_meta = load_dataset("McAuley-Lab/Amazon-Reviews-2023", "raw_meta_Cell_Phones_and_Accessories", split="full", trust_remote_code=True)
    # category = 'Basic Cases'
    # meta_cases_df, reviews_cases_df = get_category_reviews(phone_review_meta, phone_review_data, category)

    handmade_reviews_all = load_dataset("McAuley-Lab/Amazon-Reviews-2023", "raw_review_Handmade_Products",  split="full", trust_remote_code=True)
    handmade_meta_all = load_dataset("McAuley-Lab/Amazon-Reviews-2023", "raw_meta_Handmade_Products", split="full", trust_remote_code=True)

    get_multiple_reviews_csv(handmade_meta_all, handmade_reviews_all, "handmade")
    