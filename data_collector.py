import argparse
import sys
from bs4 import BeautifulSoup
from urllib import parse, request #request.urlopen
import pandas as pd

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/100.0.0.0 Safari/537.36"
}

# headers = {
#   "args": {}, 
#   "headers": {
#     "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7", 
#     "Accept-Encoding": "gzip, deflate, br, zstd", 
#     "Accept-Language": "en-US,en;q=0.9,pt;q=0.8", 
#     "Dnt": "1",  
#     "Sec-Ch-Ua": "\"Google Chrome\";v=\"123\", \"Not:A-Brand\";v=\"8\", \"Chromium\";v=\"123\"", 
#     "Sec-Ch-Ua-Mobile": "?0", 
#     "Sec-Ch-Ua-Platform": "\"macOS\"", 
#     "Sec-Fetch-Dest": "document", 
#     "Sec-Fetch-Mode": "navigate", 
#     "Sec-Fetch-Site": "cross-site", 
#     "Sec-Fetch-User": "?1", 
#     "Upgrade-Insecure-Requests": "1", 
#     "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36", 
#     "X-Amzn-Trace-Id": "Root=1-6619c6b8-3b3bca7b564e98a74928ba2c"
#   }, 
#   "origin": "128.220.159.213", 
#   "url": "https://httpbin.org/get"
# }

def get_page(url, headers):
    try:
        req = request.Request(url, headers=headers)
        with request.urlopen(req) as page: # Automatically handles closing of http connection
            soup = BeautifulSoup(page.read(), "html.parser")
    except Exception as e:
        print("Error:", e)
        return None
    
    return soup

def get_product_info(url):
    ''' Get product title, ASIN, and reviews from a product page. 
    Return dataframe with info. '''
    soup = get_page(url, headers)
    if not soup:
        print("Error: URL not found.")
        return None
    
    # Get product title
    prod_title_heading = soup.find("div", {"class": "a-row product-title"})
    prod_title = prod_title_heading.get_text().strip()
    print("product title:", prod_title)

    # Get product ASIN
    asin = soup.find("a", {"data-hook": "product-link", "class": "a-link-normal"}).get("href") # inds the first <a> tag with both data-hook and class
    asin = asin.split("/dp/")[1].split("/")[0]

    # Make dataframe with product, reviews, and stars
    df = pd.DataFrame(columns=["title", "text", "rating", "parent_asin"])

    review_content = soup.find_all("div", {"data-hook": "review"}) # get all reviews
    print("number of reviews:", len(review_content))
    print("review content:", review_content[0].prettify())
    
    new_rows = []
    for review in review_content:
        # Get review text
        review_text = review.find("span", {"class": "review-text"}).get_text().strip() # or review-body
   
        # Get star rating as float
        s = float(review.find("span", {"class": "a-icon-alt"}).get_text().strip().split(" ")[0])
        if review_text:
            new_rows.append({"title": prod_title, "text": review_text, "rating": s, "parent_asin": asin})

    new_data_rows = pd.DataFrame(new_rows)
    df = pd.concat([df, new_data_rows], ignore_index=True) # Add new rows to dataframe

    # Write data to a file
    df_csv = df.to_csv(asin+"_reviews.csv", index=False)

    return df_csv

    
# TODO: pass the reviews from get_product_info into extract_features, sentiment analyzers.

        # make better pipeline for extract_features so targeted sent analysis can be applied to custom reviews smoothly

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Scrape product reviews from Amazon")
    parser.add_argument("url", help="URL of product page on Amazon")
    args = parser.parse_args()
    url = args.url
    
    
    data = get_product_info(url) #https://www.amazon.com/Govee-Electric-Gooseneck-Temperature-Stainless/product-reviews/B09TSKDKCL/
    
