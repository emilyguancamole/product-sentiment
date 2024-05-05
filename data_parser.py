from datasets import load_dataset
import pandas as pd


def get_category_reviews(metadata, review_data, category):
    ''' Get the metadata and review data corresponding to the specified category '''

    # TODO: maybe don't write these csv, only write the processed (after stemming etc)
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

    # relevant_revs = pd.DataFrame([rev for rev in review_data if rev['parent_asin'] in relevant_asins])
    relevant_revs = []
    for i, rev in enumerate(review_data):
        if rev['parent_asin'] in relevant_asins:
            relevant_revs.append(rev)
        if i%1000==0: print(i)
        if i>10000: break # Only get first 10000 reviews
    print("relevant_revs", relevant_revs[0])
    
    relevant_revs_df = pd.DataFrame(relevant_revs)
    relevant_revs_df.to_csv(category_name + '_reviews.csv')
    return meta_df, relevant_revs_df





if __name__ == "__main__":
    '''
    Read in raw review data and metadata for a specified category, save to csv.
    Data: Amazon product review: online dataset
    '''

    # Load "full" splits of data and metadata

    review_data = load_dataset("McAuley-Lab/Amazon-Reviews-2023", "raw_review_Cell_Phones_and_Accessories", split="full", trust_remote_code=True)
    review_meta = load_dataset("McAuley-Lab/Amazon-Reviews-2023", "raw_meta_Cell_Phones_and_Accessories", split="full", trust_remote_code=True)


    category = 'Basic Cases'
    meta_cases, reviews_cases = get_category_reviews(review_meta, review_data, category)

    # Process reviews
