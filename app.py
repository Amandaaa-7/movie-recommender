import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Movie Recommender", page_icon="🎬")
st.title("🎬 Movie Recommendation System")

# =========================
# LOAD DATA
# =========================
@st.cache_data
def load_data():

    movies = pd.read_csv("movies.csv")
    ratings = pd.read_csv("ratings.csv")

    movies.columns = movies.columns.str.strip()
    ratings.columns = ratings.columns.str.strip()

    movies["title"] = movies["title"].astype(str).str.strip().str.lower()
    movies["genres"] = movies["genres"].fillna("")

    ratings = ratings.merge(movies[["movieId", "title"]], on="movieId", how="left")
    ratings = ratings.dropna(subset=["title"])

    tfidf = TfidfVectorizer(stop_words="english")
    tfidf_matrix = tfidf.fit_transform(movies["genres"])

    sim = cosine_similarity(tfidf_matrix)

    sim_df = pd.DataFrame(
        sim,
        index=movies["title"],
        columns=movies["title"]
    )

    return movies, ratings, sim_df


movies, ratings, sim_df = load_data()

movie_list = sorted(movies["title"].dropna().unique())

# =========================
# INPUT
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
# RECOMMENDER (SAFE)
# =========================
def recommend(user_input):

    watched = set([m.lower() for m, _ in user_input])
    scores = {}

    for movie, rating in user_input:

        movie = movie.lower()

        if movie not in sim_df.columns:
            continue

        for title, sim_score in sim_df[movie].items():

            if title in watched:
                continue

            # normalize score
            score = float(sim_score) * float(rating)

            scores[title] = scores.get(title, 0) + score

    if not scores:
        return pd.DataFrame()

    out = pd.DataFrame(scores.items(), columns=["title", "score"])
    out = out.sort_values("score", ascending=False).head(10)

    out = out.merge(movies[["title", "genres"]], on="title", how="left")

    return out


# =========================
# OUTPUT (NO NUMBERS SHOWN)
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
                background:#eef3ff;
                padding:12px;
                border-radius:10px;
                margin-bottom:10px;
            ">
                <h4>🎬 {row['title'].title()}</h4>
                <p>🎭 {row.get('genres','')}</p>
            </div>
            """, unsafe_allow_html=True)
