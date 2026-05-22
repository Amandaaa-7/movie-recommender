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
# LOAD DATA (SAFE)
# =========================
@st.cache_data
def load_data():

    movies = pd.read_csv("movies.csv")
    ratings = pd.read_csv("ratings.csv")

    # clean column names (CRITICAL)
    movies.columns = movies.columns.str.strip()
    ratings.columns = ratings.columns.str.strip()

    # =========================
    # DETECT MOVIE ID COLUMN
    # =========================
    def find_col(df, options):
        for c in df.columns:
            if c.lower() in options:
                return c
        return None

    movies_id = find_col(movies, ["movieid", "movie_id", "id"])
    ratings_id = find_col(ratings, ["movieid", "movie_id", "id"])

    if movies_id is None or ratings_id is None:
        st.error("Missing movieId column in one of the datasets.")
        st.stop()

    movies = movies.rename(columns={movies_id: "movieId"})
    ratings = ratings.rename(columns={ratings_id: "movieId"})

    # =========================
    # MERGE SAFE
    # =========================
    if "title" not in movies.columns:
        st.error("movies.csv must contain 'title'")
        st.stop()

    ratings = ratings.merge(
        movies[["movieId", "title"]],
        on="movieId",
        how="left"
    )

    # =========================
    # CONTENT BASED MODEL
    # =========================
    if "genres" not in movies.columns:
        movies["genres"] = ""

    tfidf = TfidfVectorizer(stop_words="english")
    tfidf_matrix = tfidf.fit_transform(movies["genres"])

    sim = cosine_similarity(tfidf_matrix)

    sim_df = pd.DataFrame(
        sim,
        index=movies["title"],
        columns=movies["title"]
    )

    # =========================
    # WEIGHTED RATING (NO POPULARITY BIAS)
    # =========================
    C = ratings["rating"].mean()
    m = ratings.groupby("movieId").size().quantile(0.60)

    stats = ratings.groupby("movieId").agg(
        v=("rating", "count"),
        R=("rating", "mean")
    ).reset_index()

    stats["weighted_rating"] = (
        (stats["v"] / (stats["v"] + m)) * stats["R"] +
        (m / (stats["v"] + m)) * C
    )

    movies = movies.merge(stats, on="movieId", how="left")
    movies["weighted_rating"] = movies["weighted_rating"].fillna(C)

    return movies, ratings, sim_df


movies, ratings, sim_df = load_data()

movie_list = movies["title"].dropna().unique()

# =========================
# USER INPUT
# =========================
st.subheader("Rate Movies")

m1 = st.selectbox("Movie 1", movie_list)
r1 = st.slider("Rating 1", 1.0, 5.0, 3.5)

m2 = st.selectbox("Movie 2", movie_list, index=1)
r2 = st.slider("Rating 2", 1.0, 5.0, 3.5)

m3 = st.selectbox("Movie 3", movie_list, index=2)
r3 = st.slider("Rating 3", 1.0, 5.0, 3.5)

user_ratings = [(m1, r1), (m2, r2), (m3, r3)]

# =========================
# RECOMMENDER ENGINE
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

            wr = movies[movies["title"] == title]["weighted_rating"].values
            wr = wr[0] if len(wr) > 0 else 3.0

            score = (sim_score * rating * 0.7) + (wr * 0.3)

            scores[title] = scores.get(title, 0) + score

    result = pd.DataFrame(scores.items(), columns=["title", "score"])
    result = result.sort_values("score", ascending=False).head(10)

    result = result.merge(
        movies[["title", "genres", "weighted_rating"]],
        on="title",
        how="left"
    )

    return result


# =========================
# OUTPUT UI (CARDS)
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
                <p><b>Popularity:</b> ⭐ {round(row['weighted_rating'], 2)}</p>
            </div>
            """, unsafe_allow_html=True)
