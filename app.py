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

    # fix column names safely
    ratings.columns = [c.lower() for c in ratings.columns]

    # EXPECTED: userId, movieId, rating
    # merge titles if needed
    if "title" not in ratings.columns:
        ratings = ratings.merge(movies[["movieId", "title"]], on="movieId", how="left")

    # fill missing
    movies["genres"] = movies["genres"].fillna("")

    # CONTENT BASED MODEL
    tfidf = TfidfVectorizer(stop_words="english")
    tfidf_matrix = tfidf.fit_transform(movies["genres"])
    sim = cosine_similarity(tfidf_matrix)

    sim_df = pd.DataFrame(sim, index=movies["title"], columns=movies["title"])

    # =========================
    # WEIGHTED RATING (NO BIAS)
    # =========================
    C = ratings["rating"].mean()
    m = ratings.groupby("movieId").size().quantile(0.60)

    movie_stats = ratings.groupby("movieId").agg(
        v=("rating", "count"),
        R=("rating", "mean")
    ).reset_index()

    movie_stats["weighted_rating"] = (
        (movie_stats["v"] / (movie_stats["v"] + m)) * movie_stats["R"] +
        (m / (movie_stats["v"] + m)) * C
    )

    movies = movies.merge(movie_stats, on="movieId", how="left")
    movies["weighted_rating"] = movies["weighted_rating"].fillna(C)

    return movies, sim_df


movies, sim_df = load_data()

movie_list = movies["title"].dropna().unique()

# =========================
# USER INPUT
# =========================
st.subheader("Rate 3 Movies")

m1 = st.selectbox("Movie 1", movie_list)
r1 = st.slider("Rating 1", 1.0, 5.0, 3.5)

m2 = st.selectbox("Movie 2", movie_list, index=1)
r2 = st.slider("Rating 2", 1.0, 5.0, 3.5)

m3 = st.selectbox("Movie 3", movie_list, index=2)
r3 = st.slider("Rating 3", 1.0, 5.0, 3.5)

user_ratings = [(m1, r1), (m2, r2), (m3, r3)]

# =========================
# HYBRID RECOMMENDER
# =========================
def recommend(user_ratings):

    watched = [m for m, _ in user_ratings]

    scores = {}

    for movie, rating in user_ratings:

        if movie not in sim_df.columns:
            continue

        for title, sim_score in sim_df[movie].items():

            if title in watched:
                continue

            # HYBRID SCORE
            weighted = movies[movies["title"] == title]["weighted_rating"].values
            weighted = weighted[0] if len(weighted) > 0 else 3.0

            score = (0.7 * sim_score * rating) + (0.3 * weighted)

            scores[title] = scores.get(title, 0) + score

    result = pd.DataFrame(scores.items(), columns=["title", "score"])
    result = result.sort_values("score", ascending=False).head(10)

    result = result.merge(movies[["title", "genres", "weighted_rating"]], on="title")

    return result


# =========================
# UI OUTPUT (CARDS)
# =========================
st.subheader("🎯 Recommendations")

if st.button("Get Recommendations"):

    recs = recommend(user_ratings)

    if recs.empty:
        st.warning("Not enough data to generate recommendations.")
    else:
        for _, row in recs.iterrows():

            st.markdown(f"""
            <div style="
                background-color:#f5f7ff;
                padding:15px;
                border-radius:12px;
                margin-bottom:10px;
                border:1px solid #ddd;
            ">
                <h4>🎬 {row['title']}</h4>
                <p><b>Genres:</b> {row['genres']}</p>
                <p><b>Score:</b> {round(row['score'], 3)}</p>
                <p><b>Popularity Score:</b> ⭐ {round(row['weighted_rating'], 2)}</p>
            </div>
            """, unsafe_allow_html=True)
