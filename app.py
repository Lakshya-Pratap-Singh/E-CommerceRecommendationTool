import pandas as pd
import numpy as np
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ================================
# LOAD DATA
# ================================

@st.cache_data
def load_data():
    df = pd.read_csv("flipkart_com-ecommerce_sample.csv")
    df = df[['product_name','description','brand','retail_price']]
    df = df.dropna().reset_index(drop=True)
    df = df.head(1000)
    return df

df = load_data()

# ================================
# SIMULATE USER RATINGS
# ================================

@st.cache_data
def simulate_users(df):

    np.random.seed(42)
    users = 100
    ratings = []

    for user in range(users):
        for _ in range(np.random.randint(5,20)):
            product = np.random.randint(0,len(df))
            rating = np.random.randint(1,6)
            ratings.append([user,product,rating])

    ratings_df = pd.DataFrame(ratings,columns=["user","product","rating"])
    return ratings_df

ratings_df = simulate_users(df)

# ================================
# COLLABORATIVE FILTERING
# ================================

@st.cache_resource
def collaborative_model():

    user_item = ratings_df.pivot_table(
        index='user',
        columns='product',
        values='rating'
    ).fillna(0)

    similarity = cosine_similarity(user_item)

    return user_item, similarity

user_item_matrix, user_similarity = collaborative_model()

# ================================
# TF-IDF CONTENT MODEL
# ================================

@st.cache_resource
def content_model():

    tfidf = TfidfVectorizer(stop_words='english')

    tfidf_matrix = tfidf.fit_transform(
        df['description'] + " " + df['brand']
    )

    similarity = cosine_similarity(tfidf_matrix)

    return similarity

content_similarity = content_model()

# ================================
# HYBRID RECOMMENDATION
# ================================

def recommend(user_id, top_n=5):

    user_scores = user_similarity[user_id]

    similar_users = np.argsort(user_scores)[::-1][1:6]

    products = ratings_df[
        ratings_df['user'].isin(similar_users)
    ]['product'].value_counts()

    recommended = products.index[:top_n]

    return df.iloc[recommended][['product_name','brand','retail_price']]

# ================================
# STREAMLIT UI
# ================================

st.title("🛒 E-Commerce Recommendation System")

st.write("Hybrid Recommendation (Collaborative + TF-IDF)")

user_id = st.number_input(
    "Enter User ID",
    min_value=0,
    max_value=99
)

if st.button("Recommend Products"):

    recs = recommend(user_id)

    st.subheader("Recommended Products")

    for _,row in recs.iterrows():

        st.markdown(f"### {row['product_name']}")
        st.write("Brand:",row['brand'])
        st.write("Price: ₹",row['retail_price'])
        st.write("---")
