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

language = st.selectbox("🌍 Language", ["English", "Arabic", "French"])

TEXT = {
    "English": {"btn": "Get Recommendations", "msg": "Top Movies", "warn": "No results"},
    "Arabic": {"btn": "احصل على التوصيات", "msg": "أفضل الأفلام", "warn": "لا توجد نتائج"},
    "French": {"btn": "Recommander", "msg": "Meilleurs films", "warn": "Aucun résultat"}
}

# =========================
# LOAD DATA
# =========================
@st.cache_data
def load_data():

    movies = pd.read_csv("movies.csv")
    ratings = pd.read_csv("ratings.csv")

    movies.columns = movies.columns.str.strip()
    ratings.columns = ratings.columns.str.strip()

    data = ratings.merge(movies, on="movieId")

    movies["genres"] = movies["genres"].fillna("")

    tfidf = TfidfVectorizer(stop_words="english")
    tfidf_matrix = tfidf.fit_transform(movies["genres"])

    sim = cosine_similarity(tfidf_matrix)

    sim_df = pd.DataFrame(
        sim,
        index=movies["title"],
        columns=movies["title"]
    )

    return movies, data, sim_df


movies, ratings, sim_df = load_data()

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
# RECOMMENDER
# =========================
def recommend(user_input):

    watched = [m for m, _ in user_input]

    # CONTENT SCORE
    content_scores = {}

    for movie in watched:
        if movie not in sim_df.columns:
            continue

        for title, score in sim_df[movie].items():
            if title in watched:
                continue
            content_scores[title] = content_scores.get(title, 0) + score

    cb = pd.DataFrame(content_scores.items(), columns=["title", "cb_score"])

    if not cb.empty:
        cb["cb_score"] = cb["cb_score"] / cb["cb_score"].max()

    # POPULARITY (simple CF proxy)
    pop = ratings.groupby("title")["rating"].mean().reset_index()
    pop.columns = ["title", "popularity"]
    pop["popularity"] = pop["popularity"] / 5.0

    pop = pop[~pop["title"].isin(watched)]

    # HYBRID
    hybrid = pd.merge(cb, pop, on="title", how="inner")

    if hybrid.empty:
        return pd.DataFrame()

    hybrid["final_score"] = 0.6 * hybrid["cb_score"] + 0.4 * hybrid["popularity"]

    hybrid = hybrid.sort_values("final_score", ascending=False).head(10)

    return hybrid.merge(movies[["title", "genres"]], on="title")


# =========================
# OUTPUT
# =========================
st.header("🎬 Recommendations")

show_table = st.checkbox("Show table view")

if st.button(TEXT[language]["btn"]):

    recs = recommend(user_input)

    if recs.empty:
        st.warning(TEXT[language]["warn"])
    else:
        st.success(TEXT[language]["msg"])

        for _, row in recs.iterrows():

            stars = "⭐" * int(round(row["popularity"] * 5))

            st.markdown(f"""
            ### 🎬 {row['title']}
            **Genres:** {row['genres']}  
            **Score:** {round(row['final_score'], 3)}  
            **Popularity:** {stars}
            ---
            """)

        if show_table:
            st.dataframe(recs)
