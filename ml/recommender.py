import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

class ProductRecommender:

    def __init__(self, df, similarity_matrix):
        self.df = df.reset_index(drop=True)
        self.similarity = similarity_matrix

        self.product_index = {
            pid: idx for idx, pid in enumerate(self.df[" Product ID"])
        }

        # --- For query based search ---
        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.text_matrix = self.vectorizer.fit_transform(
            self.df["Product Name"] + " " + self.df["Category"]
        )

    # ✅ Product ID based
    def recommend(self, product_id, top_k=5):
        if product_id not in self.product_index:
            raise ValueError("Invalid Product ID")

        idx = self.product_index[product_id]
        scores = self.similarity[idx]

        ranked = np.argsort(scores)[::-1]
        ranked = ranked[ranked != idx]

        return self.df.iloc[ranked][:top_k][[
            " Product ID",
            "Product Name",
            "Category",
            "Price",
            "Rating"
            
        ]]

    # ✅ Query based
    def recommend_from_query(self, query, top_k=5):
        query_vec = self.vectorizer.transform([query])
        similarity = cosine_similarity(query_vec, self.text_matrix)[0]

        top_indices = similarity.argsort()[::-1][:top_k]

        return self.df.iloc[top_indices][[
            " Product ID",
            "Product Name",
            "Category",
            "Price",
            "Rating"
            
        ]]