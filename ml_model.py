import numpy as np
import pandas as pd

df=pd.read_csv('Review_samsung_afterML.csv')

import streamlit as st
from sentence_transformers import SentenceTransformer

@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_model()

@st.cache_data
def load_embeddings():
    return np.load("embeddings.npy")

text_embeddings = load_embeddings()


print("Embeddings generated")
print("Shape:", text_embeddings.shape)
print("Norm check:", np.linalg.norm(text_embeddings[0]))

from sklearn.preprocessing import OneHotEncoder
encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
cat_features = encoder.fit_transform(df[["Category", "Color"]])

from sklearn.preprocessing import MinMaxScaler
num_cols = ["RAM_GB", "Storage_GB", "Price", "Rating"]
scaler = MinMaxScaler()
num_features = scaler.fit_transform(df[num_cols])


from sklearn.preprocessing import normalize
final_features = np.hstack([
    text_embeddings * 2.0,   
    cat_features * 1.0,      
    num_features * 0.5       
])
final_features = normalize(final_features)


from sklearn.metrics.pairwise import cosine_similarity
cosine_sim_matrix = cosine_similarity(final_features)


class ProductRecommender:
    def __init__(self, dataframe, similarity_matrix):
        self.df = dataframe.reset_index(drop=True)
        self.similarity = similarity_matrix
        self.product_index = {
            pid: idx for idx, pid in enumerate(self.df[" Product ID"])
        }
        

    def recommend(self, product_id, top_k=5):
        if product_id not in self.product_index:
            raise ValueError("Invalid Product ID")

        idx = self.product_index[product_id]
        scores = self.similarity[idx]

        ranked_indices = np.argsort(scores)[::-1]
        ranked_indices = ranked_indices[ranked_indices != idx]

        return self.df.iloc[ranked_indices][:top_k][[
            " Product ID",
            "Product Name",
            "Category",
            "Price",
            "Rating",
            "Image URL"
        ]]
    
    def recommend_from_query(self, query, top_k=5):
        query_embedding = model.encode(
        [query],
        normalize_embeddings=True
        )

        query_vector = np.hstack([
        query_embedding * 2.0,
        np.zeros((1, cat_features.shape[1])),
        np.zeros((1, num_features.shape[1]))
        ])

        query_vector = normalize(query_vector)

        similarity = cosine_similarity(query_vector, final_features)[0]
        top_indices = similarity.argsort()[::-1][:top_k]

        return df.iloc[top_indices][[
            " Product ID",
            "Product Name",
            "Category",
            "Price",
            "Rating",
            "Image URL"
        ]]


recommender = ProductRecommender(df, cosine_sim_matrix)
print(recommender.recommend("WAT_005", top_k=5))
print(recommender.recommend_from_query("LTE smartwatch with health tracking", 5))

print(recommender.recommend_from_query("split air conditioner", 10))
print(recommender.recommend("TEL_050", top_k=10))

#evaluation metrics
def category_consistency(df, sim_matrix, top_k=10):
    matches = []
    for i in range(len(df)):
        top_indices = np.argsort(sim_matrix[i])[::-1][1: top_k+1]
        base_cat = df.iloc[i]["Category"]
        rec_cats = df.iloc[top_indices]["Category"]
        matches.append((rec_cats == base_cat).mean())
    return np.mean(matches)
def average_similarity(sim_matrix, top_k=10):
    scores = []
    for i in range(sim_matrix.shape[0]):
        sims = sim_matrix[i]
        top_indices = np.argsort(sims)[::-1][1: top_k+1]
        scores.append(sims[top_indices].mean())

    return np.mean(scores)

cat_score = category_consistency(df, cosine_sim_matrix)
print("Category Consistency:", cat_score)
avg_sim = average_similarity(cosine_sim_matrix, top_k=5)
print("Average Similarity :",avg_sim)
