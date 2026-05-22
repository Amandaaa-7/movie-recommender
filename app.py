import streamlit as st
import pandas as pd
import numpy as np
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

    movies["genres"] = movies["genres"].fillna("")

    # =========================
    # CONTENT-BASED MODEL
    # =========================
    tfidf = TfidfVectorizer(stop_words="english")
    tfidf_matrix = tfidf.fit_transform(movies["genres"])

    similarity = cosine_similarity(tfidf_matrix)

    sim_df = pd.DataFrame(
        similarity,
        index=movies["title"],
        columns=movies["title"]
    )

    # =========================
    # COLLAB FILTER (SAFE VERSION)
    # =========================
    cf_scores = ratings.groupby("title")["rating"].mean().to_dict()

    return movies, ratings, sim_df, cf_scores


movies, ratings, sim_df, cf_scores = load_data()

movie_list = movies["title"].dropna().unique()

# =========================
# USER INPUT
# =========================
st.header("Rate Movies You Like")

m1 = st.selectbox("Movie 1", movie_list)
r1 = st.slider("Rating 1", 0.5, 5.0, 4.0)

m2 = st.selectbox("Movie 2", movie_list, index=1)
r2 = st.slider("Rating 2", 0.5, 5.0, 4.0)

m3 = st.selectbox("Movie 3", movie_list, index=2)
r3 = st.slider("Rating 3", 0.5, 5.0, 4.0)

user_input = [(m1, r1), (m2, r2), (m3, r3)]

# =========================
# RECOMMENDER FUNCTION
# =========================
def recommend(user_input):

    watched = [m for m, _ in user_input]

    # =========================
    # CONTENT-BASED SCORE
    # =========================
    content_scores = {}

    for movie in watched:
        if movie not in sim_df.columns:
            continue

        for title, score in sim_df[movie].items():
            if title in watched:
                continue

            content_scores[title] = content_scores.get(title, 0) + score

    cb_df = pd.DataFrame(content_scores.items(), columns=["title", "cb_score"])

    # =========================
    # COLLAB FILTER SCORE (SAFE)
    # =========================
    cf_df = pd.DataFrame([
        (title, cf_scores.get(title, 3.0))
        for title in cb_df["title"]
    ], columns=["title", "cf_score"])

    # =========================
    # MERGE
    # =========================
    hybrid = pd.merge(cb_df, cf_df, on="title", how="inner")

    if hybrid.empty:
        return pd.DataFrame()

    # =========================
    # NORMALIZATION
    # =========================
    hybrid["cb_score"] = (
        hybrid["cb_score"] - hybrid["cb_score"].min()
    ) / (hybrid["cb_score"].max() - hybrid["cb_score"].min() + 1e-9)

    hybrid["cf_score"] = hybrid["cf_score"] / 5.0

    # =========================
    # FINAL HYBRID SCORE
    # =========================
    hybrid["final_score"] = (
        0.6 * hybrid["cb_score"] +
        0.4 * hybrid["cf_score"]
    )

    hybrid = hybrid.sort_values("final_score", ascending=False).head(10)

    # =========================
    # ADD GENRES (IMPORTANT)
    # =========================
    result = pd.merge(
        hybrid,
        movies[["title", "genres"]],
        on="title",
        how="left"
    )

    return result


# =========================
# OUTPUT
# =========================
st.header("Top Recommendations")

if st.button("Get Recommendations"):

    recs = recommend(user_input)

    if recs.empty:
        st.warning("Not enough data to generate recommendations.")
    else:
        st.dataframe(
            recs.rename(columns={
                "title": "Movie",
                "genres": "Genres",
                "final_score": "Score"
            })
        )
