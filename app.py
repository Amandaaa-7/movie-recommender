import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="Hybrid Movie Recommender",
    page_icon="🎬",
    layout="centered"
)

st.title("🎬 Hybrid Movie Recommendation System")

# =========================
# LOAD DATA
# =========================
@st.cache_data
def load_data():

    movies = pd.read_csv("clean_content.csv")
    ratings = pd.read_csv("ratings.csv")

    # Clean
    movies["genres"] = movies["genres"].fillna("")

    # CONTENT MODEL
    tfidf = TfidfVectorizer(stop_words="english")
    tfidf_matrix = tfidf.fit_transform(movies["genres"])
    similarity = cosine_similarity(tfidf_matrix)

    sim_df = pd.DataFrame(
        similarity,
        index=movies["title"],
        columns=movies["title"]
    )

    return movies, ratings, sim_df

movies, ratings, sim_df = load_data()

# =========================
# LOAD SVD MODEL (IMPORTANT)
# =========================
svd_model = joblib.load("svd_model.pkl")

# =========================
# USER INPUT
# =========================
movie_list = movies["title"].dropna().unique()

st.header("Select Movies You Like")

m1 = st.selectbox("Movie 1", movie_list)
r1 = st.slider("Rating 1", 0.5, 5.0, 3.5)

m2 = st.selectbox("Movie 2", movie_list, index=1)
r2 = st.slider("Rating 2", 0.5, 5.0, 3.5)

m3 = st.selectbox("Movie 3", movie_list, index=2)
r3 = st.slider("Rating 3", 0.5, 5.0, 3.5)

user_ratings = [(m1, r1), (m2, r2), (m3, r3)]

# =========================
# HYBRID FUNCTION
# =========================
def recommend(user_ratings):

    watched = [m for m, _ in user_ratings]

    # ---------- CONTENT BASED ----------
    content_scores = {}

    for movie in watched:
        if movie not in sim_df.columns:
            continue

        for title, score in sim_df[movie].items():
            if title in watched:
                continue

            content_scores[title] = content_scores.get(title, 0) + score

    content_df = pd.DataFrame(content_scores.items(), columns=["title", "cb_score"])

    # ---------- COLLABORATIVE (SVD REAL) ----------
    cf_scores = {}

    for _, row in ratings.iterrows():
        if row["title"] in watched:
            continue

        try:
            pred = svd_model.predict(row["userId"], row["movieId"]).est
            cf_scores[row["title"]] = pred
        except:
            continue

    cf_df = pd.DataFrame(cf_scores.items(), columns=["title", "cf_score"])

    # ---------- MERGE ----------
    hybrid = pd.merge(content_df, cf_df, on="title", how="inner")

    if hybrid.empty:
        return pd.DataFrame()

    # normalize
    hybrid["cb_score"] = (hybrid["cb_score"] - hybrid["cb_score"].min()) / (hybrid["cb_score"].max() + 1e-9)
    hybrid["cf_score"] = hybrid["cf_score"] / 5.0   # MovieLens scale fix

    hybrid["final_score"] = 0.5 * hybrid["cb_score"] + 0.5 * hybrid["cf_score"]

    hybrid = hybrid.sort_values("final_score", ascending=False).head(10)

    return hybrid

# =========================
# OUTPUT
# =========================
st.header("Recommendations")

if st.button("Get Recommendations"):

    recs = recommend(user_ratings)

    if recs.empty:
        st.warning("Not enough data to generate recommendations.")
    else:
        st.dataframe(recs)
