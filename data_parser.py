from datasets import load_dataset

# data_beauty = load_dataset("McAuley-Lab/Amazon-Reviews-2023", "raw_review_All_Beauty", trust_remote_code=True)
# print(data_beauty["full"][1]) # print the second review in the dataset

data_cellphone = load_dataset("McAuley-Lab/Amazon-Reviews-2023", "raw_review_Cell_Phones_and_Accessories", trust_remote_code=True)
meta_cellphone = load_dataset("McAuley-Lab/Amazon-Reviews-2023", "raw_meta_Cell_Phones_and_Accessories", split="full", trust_remote_code=True)


print(data_cellphone["full"][1]) 
print(meta_cellphone[0])
# POS tagger, pull out all noun sequences, make histogram of occurrences, take top k (remove junk)

