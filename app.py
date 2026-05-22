import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# =========================
# CONFIG
# =========================
st.set_page_config(page_title="Movie Recommender", page_icon="🎬", layout="centered")
st.title("🎬 Movie Recommendation System")

# =========================
# LOAD DATA (SAFE + FAST)
# =========================
@st.cache_data
def load_data():

    movies = pd.read_csv("movies.csv")
    ratings = pd.read_csv("ratings.csv")

    movies.columns = movies.columns.str.strip()
    ratings.columns = ratings.columns.str.strip()

    # normalize titles (CRITICAL FIX)
    movies["title"] = movies["title"].astype(str).str.strip().str.lower()
    movies["genres"] = movies["genres"].fillna("")

    # safe merge
    ratings = ratings.merge(movies[["movieId", "title"]], on="movieId", how="left")
    ratings = ratings.dropna(subset=["title"])

    # LIMIT SIZE (PREVENT CRASH)
    movies = movies.head(5000).copy()

    # CONTENT MODEL
    tfidf = TfidfVectorizer(stop_words="english")
    tfidf_matrix = tfidf.fit_transform(movies["genres"])

    return movies, ratings, tfidf_matrix


movies, ratings, tfidf_matrix = load_data()

movie_list = sorted(movies["title"].unique())

# =========================
# USER INPUT
# =========================
st.subheader("Rate Movies")

m1 = st.selectbox("Movie 1", movie_list)
r1 = st.slider("Rating 1", 1.0, 5.0, 4.0)

m2 = st.selectbox("Movie 2", movie_list, index=1)
r2 = st.slider("Rating 2", 1.0, 5.0, 4.0)

m3 = st.selectbox("Movie 3", movie_list, index=2)
r3 = st.slider("Rating 3", 1.0, 5.0, 4.0)

user_input = [(m1, r1), (m2, r2), (m3, r3)]

# =========================
# RECOMMENDER (LAZY SAFE VERSION)
# =========================
def recommend(user_input):

    watched = set([m.lower() for m, _ in user_input])
    scores = {}

    for movie, rating in user_input:

        movie = movie.lower()

        if movie not in movies["title"].values:
            continue

        idx = movies[movies["title"] == movie].index[0]

        # compute similarity ONLY for this movie (NOT full matrix)
        sim_scores = cosine_similarity(tfidf_matrix[idx], tfidf_matrix).flatten()

        for i, score in enumerate(sim_scores):

            title = movies.iloc[i]["title"]

            if title in watched:
                continue

            # safe weighted score
            final_score = float(score) * float(rating)

            scores[title] = scores.get(title, 0) + final_score

    if not scores:
        return pd.DataFrame()

    recs = pd.DataFrame(scores.items(), columns=["title", "score"])
    recs = recs.sort_values("score", ascending=False).head(10)

    recs = recs.merge(movies[["title", "genres"]], on="title", how="left")

    return recs


# =========================
# OUTPUT (NO NUMBERS SHOWN)
# =========================
st.subheader("🎬 Recommendations")

if st.button("Get Recommendations"):

    with st.spinner("Finding movies..."):
        recs = recommend(user_input)

    if recs is None or recs.empty:
        st.warning("No recommendations found.")
    else:
        for _, row in recs.iterrows():

            html = (
                "<div style='"
                "background:#eef3ff;"
                "padding:12px;"
                "border-radius:12px;"
                "margin-bottom:10px;"
                "color:black;"
                "'>"
                f"<h4 style='margin:0; color:black;'>🎬 {row['title'].title()}</h4>"
                f"<p style='margin:5px 0 0 0; color:black;'>🎭 {row.get('genres','')}</p>"
                "</div>"
            )

            st.markdown(html, unsafe_allow_html=True)
