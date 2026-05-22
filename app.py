import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# =========================
# CONFIG
# =========================
st.set_page_config(page_title="Movie Recommender", page_icon="🎬", layout="centered")
st.title("🎬 Hybrid Movie Recommender")

# =========================
# SAFE LOAD
# =========================
@st.cache_data
def load_data():

    movies = pd.read_csv("movies.csv")
    ratings = pd.read_csv("ratings.csv")

    # clean columns
    movies.columns = movies.columns.str.strip().str.lower()
    ratings.columns = ratings.columns.str.strip().str.lower()

    # normalize column names
    movies = movies.rename(columns={
        "movieid": "movieId",
        "movie_id": "movieId",
        "title ": "title"
    })

    ratings = ratings.rename(columns={
        "movieid": "movieId",
        "movie_id": "movieId"
    })

    # safety checks
    if "movieId" not in movies.columns or "movieId" not in ratings.columns:
        st.error("Missing movieId column in dataset")
        st.stop()

    if "title" not in movies.columns:
        st.error("Missing title column in movies.csv")
        st.stop()

    if "genres" not in movies.columns:
        movies["genres"] = ""

    # clean text
    movies["title"] = movies["title"].astype(str).str.strip().str.lower()
    movies["genres"] = movies["genres"].fillna("")

    ratings = ratings.merge(movies[["movieId", "title"]], on="movieId", how="left")
    ratings = ratings.dropna(subset=["title"])

    # =========================
    # CONTENT MODEL (SAFE SMALL SCALE)
    # =========================
    tfidf = TfidfVectorizer(stop_words="english")

    try:
        tfidf_matrix = tfidf.fit_transform(movies["genres"])
        sim = cosine_similarity(tfidf_matrix)
    except:
        st.error("Content model failed")
        st.stop()

    sim_df = pd.DataFrame(sim, index=movies["title"], columns=movies["title"])

    # =========================
    # POPULARITY (CF PROXY)
    # =========================
    pop = ratings.groupby("title")["rating"].mean().reset_index()
    pop.columns = ["title", "popularity"]
    pop["popularity"] = pop["popularity"] / 5.0

    return movies, sim_df, pop


movies, sim_df, pop = load_data()

movie_list = sorted(movies["title"].dropna().unique())

# =========================
# USER INPUT
# =========================
st.subheader("Rate Movies")

m1 = st.selectbox("Movie 1", movie_list)
m2 = st.selectbox("Movie 2", movie_list, index=1)
m3 = st.selectbox("Movie 3", movie_list, index=2)

user_input = [m1, m2, m3]

# =========================
# RECOMMENDER
# =========================
def recommend(user_input):

    watched = set(user_input)
    scores = {}

    for movie in user_input:

        if movie not in sim_df.columns:
            continue

        for title, sim_score in sim_df[movie].items():

            if title in watched:
                continue

            pop_score = pop.loc[pop["title"] == title, "popularity"]

            if len(pop_score) == 0:
                pop_score = 0.5
            else:
                pop_score = float(pop_score.values[0])

            score = sim_score + 0.3 * pop_score

            scores[title] = scores.get(title, 0) + score

    if not scores:
        return pd.DataFrame()

    out = pd.DataFrame(scores.items(), columns=["title", "score"])
    out = out.sort_values("score", ascending=False).head(10)

    out = out.merge(movies[["title", "genres"]], on="title", how="left")

    return out


# =========================
# OUTPUT (NO SCORES SHOWN)
# =========================
st.subheader("🎬 Recommendations")

if st.button("Get Recommendations"):

    recs = recommend(user_input)

    if recs.empty:
        st.warning("No recommendations found")
    else:
        for _, row in recs.iterrows():

            st.markdown(f"""
            <div style="
                background-color:#eef3ff;
                padding:12px;
                border-radius:10px;
                margin-bottom:10px;
            ">
                <h4>🎬 {row['title'].title()}</h4>
                <p>🎭 {row.get('genres','')}</p>
            </div>
            """, unsafe_allow_html=True)
