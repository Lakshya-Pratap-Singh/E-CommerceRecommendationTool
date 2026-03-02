# ============================================
# HYBRID E-COMMERCE RECOMMENDATION SYSTEM
# (TF-IDF + SVD) - STREAMLIT CLOUD SAFE
# ============================================

import pandas as pd
import numpy as np
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from surprise import Dataset, Reader, SVD

# ============================================
# LOAD DATASET
# ============================================

@st.cache_data
def load_data():
    df = pd.read_csv("flipkart_com-ecommerce_sample.csv")
    df = df[['product_name', 'description', 'brand', 'retail_price']]
    df = df.dropna().reset_index(drop=True)
    df = df.head(1000)  # limit for faster deployment
    return df

df = load_data()

# ============================================
# SIMULATE USER RATINGS
# ============================================

@st.cache_data
def generate_ratings(df):
    np.random.seed(42)
    num_users = 100
    ratings_data = []

    for user_id in range(num_users):
        for _ in range(np.random.randint(5, 20)):
            product_id = np.random.randint(0, len(df))
            rating = np.random.randint(1, 6)
            ratings_data.append([user_id, product_id, rating])

    return pd.DataFrame(ratings_data, columns=["user_id", "product_id", "rating"])

ratings_df = generate_ratings(df)

# ============================================
# TRAIN COLLABORATIVE MODEL (SVD)
# ============================================

@st.cache_resource
def train_svd(ratings_df):
    reader = Reader(rating_scale=(1, 5))
    data = Dataset.load_from_df(ratings_df[['user_id','product_id','rating']], reader)
    trainset = data.build_full_trainset()
    model = SVD()
    model.fit(trainset)
    return model

svd_model = train_svd(ratings_df)

# ============================================
# TF-IDF CONTENT MODEL
# ============================================

@st.cache_resource
def compute_tfidf(df):
    tfidf = TfidfVectorizer(stop_words='english', max_features=5000)
    tfidf_matrix = tfidf.fit_transform(
        df['description'].astype(str) + " " + df['brand'].astype(str)
    )
    similarity = cosine_similarity(tfidf_matrix)
    return similarity

similarity_matrix = compute_tfidf(df)

# ============================================
# HYBRID RECOMMENDATION FUNCTION
# ============================================

def hybrid_recommend(user_id, top_n=5):

    scores = []

    user_high_rated = ratings_df[
        (ratings_df['user_id'] == user_id) &
        (ratings_df['rating'] >= 4)
    ]['product_id'].values

    for product_id in range(len(df)):

        # Collaborative Score
        collab_score = svd_model.predict(user_id, product_id).est

        # Content Score
        if len(user_high_rated) > 0:
            content_score = np.mean(
                [similarity_matrix[product_id][i] for i in user_high_rated]
            )
        else:
            content_score = 0

        # Final Hybrid Score
        final_score = (0.6 * collab_score) + (0.4 * content_score)

        scores.append((product_id, final_score))

    scores = sorted(scores, key=lambda x: x[1], reverse=True)
    recommended_ids = [i[0] for i in scores[:top_n]]

    return df.iloc[recommended_ids][['product_name','brand','retail_price']]

# ============================================
# STREAMLIT UI
# ============================================

st.title("🛒 Hybrid E-Commerce Recommendation System")
st.write("TF-IDF + Collaborative Filtering (SVD)")

user_id = st.number_input(
    "Enter User ID (0 - 99)",
    min_value=0,
    max_value=99,
    step=1
)

if st.button("Get Recommendations"):

    results = hybrid_recommend(user_id)

    st.subheader("Recommended Products:")

    for index, row in results.iterrows():
        st.markdown(f"### {row['product_name']}")
        st.write(f"Brand: {row['brand']}")
        st.write(f"Price: ₹{row['retail_price']}")
        st.write("---")
