import pandas as pd

# test = pd.read_pickle("./rag_search/data/text_data.pkl")
test = pd.read_pickle("./rag_search/data/image_data.pkl")
print(max([item.shape[1] for item in test['image_vector']]))