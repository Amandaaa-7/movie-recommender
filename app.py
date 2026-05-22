import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# =========================
# CONFIG
# =========================
st.set_page_config(page_title="Hybrid Movie Recommender", page_icon="🎬")

st.title("🎬 Production-Grade Hybrid Movie Recommender")

# =========================
# SAFE DATA LOADER
# =========================
@st.cache_data
def load_data():

    movies = pd.read_csv("movies.csv")
    ratings = pd.read_csv("ratings.csv")

    # Clean column names
    movies.columns = movies.columns.str.strip()
    ratings.columns = ratings.columns.str.strip()

    # Normalize text fields
    if "title" in movies.columns:
        movies["title"] = movies["title"].astype(str).str.strip()

    # Detect ID columns safely
    def detect_id(df):
        for c in df.columns:
            if c.lower() in ["movieid", "movie_id", "id"]:
                return c
        return None

    m_id = detect_id(movies)
    r_id = detect_id(ratings)

    if m_id is None or r_id is None:
        st.error("Missing movie ID column in dataset")
        st.stop()

    movies = movies.rename(columns={m_id: "movieId"})
    ratings = ratings.rename(columns={r_id: "movieId"})

    # Ensure required columns exist
    if "title" not in movies.columns:
        st.error("movies.csv must contain 'title'")
        st.stop()

    if "genres" not in movies.columns:
        movies["genres"] = ""

    # Merge safely
    ratings = ratings.merge(
        movies[["movieId", "title"]],
        on="movieId",
        how="left"
    )

    ratings = ratings.dropna(subset=["title"])

    # =========================
    # CONTENT MODEL
    # =========================
    tfidf = TfidfVectorizer(stop_words="english")
    tfidf_matrix = tfidf.fit_transform(movies["genres"].fillna(""))

    sim = cosine_similarity(tfidf_matrix)

    sim_df = pd.DataFrame(
        sim,
        index=movies["title"],
        columns=movies["title"]
    )

    # Clean similarity matrix labels
    sim_df.index = sim_df.index.str.strip()
    sim_df.columns = sim_df.columns.str.strip()

    # =========================
    # WEIGHTED POPULARITY (NO BIAS)
    # =========================
    C = ratings["rating"].mean()
    m = ratings.groupby("movieId").size().quantile(0.6)

    stats = ratings.groupby("movieId").agg(
        count=("rating", "count"),
        mean=("rating", "mean")
    ).reset_index()

    stats["weighted"] = (
        (stats["count"] / (stats["count"] + m)) * stats["mean"]
        + (m / (stats["count"] + m)) * C
    )

    movies = movies.merge(stats[["movieId", "weighted"]], on="movieId", how="left")
    movies["weighted"] = movies["weighted"].fillna(C)

    return movies, sim_df


movies, sim_df = load_data()

movie_list = sorted(movies["title"].dropna().unique())

# =========================
# USER INPUT
# =========================
st.subheader("Rate Movies")

def safe_index(i):
    return min(i, len(movie_list)-1)

m1 = st.selectbox("Movie 1", movie_list)
r1 = st.slider("Rating 1", 1.0, 5.0, 3.5)

m2 = st.selectbox("Movie 2", movie_list, index=safe_index(1))
r2 = st.slider("Rating 2", 1.0, 5.0, 3.5)

m3 = st.selectbox("Movie 3", movie_list, index=safe_index(2))
r3 = st.slider("Rating 3", 1.0, 5.0, 3.5)

user_input = [(m1, r1), (m2, r2), (m3, r3)]

# =========================
# SAFE RECOMMENDER
# =========================
def recommend(user_input):

    watched = set([m for m, _ in user_input])
    scores = {}

    for movie, rating in user_input:

        if movie not in sim_df.columns:
            continue

        try:
            for title, sim_score in sim_df[movie].items():

                if title in watched:
                    continue

                w = movies.loc[movies["title"] == title, "weighted"]

                if len(w) == 0:
                    weighted = 3.0
                else:
                    weighted = float(w.values[0])

                score = sim_score * rating + weighted * 0.3

                scores[title] = scores.get(title, 0) + score

        except Exception:
            continue

    if not scores:
        return pd.DataFrame(columns=["title", "score"])

    out = pd.DataFrame(scores.items(), columns=["title", "score"])
    out = out.sort_values("score", ascending=False).head(10)

    out = out.merge(movies[["title", "genres", "weighted"]], on="title", how="left")

    return out


# =========================
# OUTPUT UI (CARDS)
# =========================
st.subheader("🎯 Recommendations")

if st.button("Get Recommendations"):

    with st.spinner("Generating recommendations..."):
        recs = recommend(user_input)

    if recs.empty:
        st.warning("Not enough data to generate recommendations.")
    else:
        for _, row in recs.iterrows():

            st.markdown(f"""
            <div style="
                background-color:#eef3ff;
                padding:12px;
                border-radius:10px;
                margin-bottom:10px;
            ">
                <h4>🎬 {row['title']}</h4>
                <p>🎭 {row.get('genres','')}</p>
                <p>⭐ Score: {round(row['score'], 3)}</p>
                <p>📊 Weighted: {round(row['weighted'], 2)}</p>
            </div>
            """, unsafe_allow_html=True)
