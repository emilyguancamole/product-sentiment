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
    soup = get_page(url, headers)
    if not soup:
        print("WELP")
        return None, None, None
    
    # Get product title
    # prod_title_heading = soup.find("h1", {"id": "title"})
    prod_title_heading = soup.find("div", {"class": "a-row product-title"})
    prod_title = prod_title_heading.get_text().strip()
    print("product title:", prod_title)

    # todo: Get product ASIN

    # Make dataframe with product, reviews, and stars
    df = pd.DataFrame(columns=["product_title", "review", "star_rating"])

    review_content = soup.find_all("div", {"data-hook": "review"}) # get all reviews
    new_rows = []
    for review in review_content:
        # get all review content
        # for r in review.find_all("span", {"class": "review-text"}):
        #     review_text = r.get_text()
        #     print(review_text) 
        review_text = review.find("span", {"class": "review-text"}).get_text().strip() # or review-body
        print(review_text)
        # get all review stars
        s = review.find("span", {"class": "a-icon-alt"}).get_text().strip()
        print(s)
        
        new_rows.append({"product_title": prod_title, "review": review_text, "star_rating": s})

    new_data_rows = pd.DataFrame(new_rows)
    df = pd.concat([df, new_data_rows], ignore_index=True) # Add new rows to dataframe
    # Write data to a file
    df_csv = df.to_csv("product_reviews.csv", index=False)

    return df_csv

    

if __name__ == "__main__":
    url = "https://www.amazon.com/Govee-Electric-Gooseneck-Temperature-Stainless/product-reviews/B09TSKDKCL/" # reviews page
    # url = "https://www.amazon.com/dp/B0BQBMYR5R/"
    data = get_product_info(url)
    
