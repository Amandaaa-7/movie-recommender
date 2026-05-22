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

# =========================
# TITLE
# =========================
st.title("🎬 Hybrid Movie Recommendation System")

st.write("Content-based + Collaborative filtering (clean deployment version)")

# =========================
# LANGUAGE SUPPORT
# =========================
language = st.selectbox("🌍 Language", ["English", "Arabic", "French"])

TEXT = {
    "English": {
        "btn": "Get Recommendations",
        "msg": "Top Recommendations",
        "warn": "Not enough data"
    },
    "Arabic": {
        "btn": "الحصول على التوصيات",
        "msg": "أفضل الأفلام المقترحة",
        "warn": "لا توجد بيانات كافية"
    },
    "French": {
        "btn": "Obtenir des recommandations",
        "msg": "Meilleurs films recommandés",
        "warn": "Pas assez de données"
    }
}

# =========================
# LOAD DATA
# =========================
@st.cache_data
def load_data():

    movies = pd.read_csv("movies.csv")
    ratings = pd.read_csv("ratings.csv")

    # clean columns
    movies.columns = movies.columns.str.strip()
    ratings.columns = ratings.columns.str.strip()

    # merge
    data = ratings.merge(movies, on="movieId")

    # content preprocessing
    movies["genres"] = movies["genres"].fillna("")

    tfidf = TfidfVectorizer(stop_words="english")
    tfidf_matrix = tfidf.fit_transform(movies["genres"])

    similarity = cosine_similarity(tfidf_matrix)

    sim_df = pd.DataFrame(
        similarity,
        index=movies["title"],
        columns=movies["title"]
    )

    # collaborative filtering proxy (stable version)
    cf_scores = data.groupby("movieId")["rating"].mean().to_dict()

    return movies, data, sim_df, cf_scores


movies, ratings, sim_df, cf_scores = load_data()

movie_list = sorted(movies["title"].dropna().unique())

# =========================
# USER INPUT
# =========================
st.header("🎯 Rate Movies")

m1 = st.selectbox("Movie 1", movie_list)
r1 = st.slider("Rating 1", 1.0, 5.0, 4.0)

m2 = st.selectbox("Movie 2", movie_list, index=1)
r2 = st.slider("Rating 2", 1.0, 5.0, 4.0)

m3 = st.selectbox("Movie 3", movie_list, index=2)
r3 = st.slider("Rating 3", 1.0, 5.0, 4.0)

user_input = [(m1, r1), (m2, r2), (m3, r3)]

# =========================
# RECOMMENDER FUNCTION
# =========================
def recommend(user_input):

    watched = [m for m, _ in user_input]

    # ---------------- CONTENT BASED ----------------
    content_scores = {}

    for movie in watched:
        if movie not in sim_df.columns:
            continue

        for title, score in sim_df[movie].items():
            if title in watched:
                continue

            content_scores[title] = content_scores.get(title, 0) + score

    cb_df = pd.DataFrame(content_scores.items(), columns=["title", "cb_score"])

    # normalize
    if not cb_df.empty:
        cb_df["cb_score"] = cb_df["cb_score"] / cb_df["cb_score"].max()

    # ---------------- COLLAB FILTER (SAFE) ----------------
    cf_df = ratings.groupby("title")["rating"].mean().reset_index()
    cf_df.columns = ["title", "cf_score"]

    cf_df = cf_df[~cf_df["title"].isin(watched)]

    if not cf_df.empty:
        cf_df["cf_score"] = cf_df["cf_score"] / 5.0  # FIX SCALE

    # ---------------- HYBRID ----------------
    hybrid = pd.merge(cb_df, cf_df, on="title", how="inner")

    if hybrid.empty:
        return pd.DataFrame()

    hybrid["final_score"] = (
        0.5 * hybrid["cb_score"] +
        0.5 * hybrid["cf_score"]
    )

    hybrid = hybrid.sort_values("final_score", ascending=False).head(10)

    result = hybrid.merge(movies[["title", "genres"]], on="title")

    return result


# =========================
# OUTPUT
# =========================
st.header("🎬 Recommendations")

if st.button(TEXT[language]["btn"]):

    with st.spinner("Processing..."):
        recs = recommend(user_input)

    if recs.empty:
        st.warning(TEXT[language]["warn"])
    else:
        st.success(TEXT[language]["msg"])

        st.dataframe(
            recs[["title", "genres", "final_score"]].rename(columns={
                "title": "Movie",
                "genres": "Genres",
                "final_score": "Score"
            })
        )
