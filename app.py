import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# =========================
# CONFIG
# =========================
st.set_page_config(page_title="Hybrid Movie Recommender", page_icon="🎬")
st.title("🎬 Hybrid Movie Recommender")

# =========================
# LOAD DATA (SAFE)
# =========================
@st.cache_data
def load_data():

    movies = pd.read_csv("movies.csv")
    ratings = pd.read_csv("ratings.csv")

    # clean columns
    movies.columns = movies.columns.str.strip().str.lower()
    ratings.columns = ratings.columns.str.strip().str.lower()

    # normalize IDs
    movies = movies.rename(columns={"movieid": "movieId"})
    ratings = ratings.rename(columns={"movieid": "movieId"})

    # safety checks
    if "movieId" not in movies.columns or "movieId" not in ratings.columns:
        st.error("Missing movieId column")
        st.stop()

    if "title" not in movies.columns:
        st.error("Missing title column")
        st.stop()

    movies["title"] = movies["title"].astype(str).str.strip().str.lower()
    movies["genres"] = movies.get("genres", "").fillna("")

    # merge ratings
    ratings = ratings.merge(movies[["movieId", "title"]], on="movieId", how="left")
    ratings = ratings.dropna(subset=["title"])

    # =========================
    # CONTENT MODEL
    # =========================
    tfidf = TfidfVectorizer(stop_words="english")
    tfidf_matrix = tfidf.fit_transform(movies["genres"])

    sim = cosine_similarity(tfidf_matrix)

    sim_df = pd.DataFrame(sim, index=movies["title"], columns=movies["title"])

    # popularity proxy
    pop = ratings.groupby("title")["rating"].mean().reset_index()
    pop.columns = ["title", "popularity"]
    pop["popularity"] = pop["popularity"] / 5.0

    return movies, sim_df, pop


movies, sim_df, pop = load_data()

movie_list = sorted(movies["title"].dropna().unique())

# =========================
# USER INPUT (SLIDERS RESTORED)
# =========================
st.subheader("Rate Movies")

m1 = st.selectbox("Movie 1", movie_list)
r1 = st.slider("Rating 1", 1.0, 5.0, 3.0)

m2 = st.selectbox("Movie 2", movie_list, index=1)
r2 = st.slider("Rating 2", 1.0, 5.0, 3.0)

m3 = st.selectbox("Movie 3", movie_list, index=2)
r3 = st.slider("Rating 3", 1.0, 5.0, 3.0)

user_input = [(m1, r1), (m2, r2), (m3, r3)]

# =========================
# RECOMMENDER (CRASH SAFE)
# =========================
def recommend(user_input):

    watched = set([m for m, _ in user_input])
    scores = {}

    for movie, rating in user_input:

        if movie not in sim_df.columns:
            continue

        for title, sim_score in sim_df[movie].items():

            if title in watched:
                continue

            if title not in pop["title"].values:
                pop_score = 0.5
            else:
                pop_score = float(pop.loc[pop["title"] == title, "popularity"].values[0])

            score = sim_score * rating + 0.2 * pop_score
            scores[title] = scores.get(title, 0) + score

    if not scores:
        return pd.DataFrame()

    out = pd.DataFrame(scores.items(), columns=["title", "score"])
    out = out.sort_values("score", ascending=False).head(10)

    return out.merge(movies[["title", "genres"]], on="title", how="left")


# =========================
# OUTPUT (NO CRASH)
# =========================
st.subheader("🎬 Recommendations")

if st.button("Get Recommendations"):

    try:
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

    except Exception as e:
        st.error("App crashed safely (debug mode):")
        st.exception(e)
