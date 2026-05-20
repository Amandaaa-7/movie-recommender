import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---------------- UI ----------------
st.set_page_config(page_title="Movie Recommender", page_icon="🎬")

st.title("🎬 Hybrid Movie Recommendation System")
st.write("Select movies you like and get recommendations.")

# ---------------- LOAD DATA ----------------
@st.cache_data
def load_data():

    movies = pd.read_csv("clean_content.csv")

    ratings = pd.read_csv(
        "ratings_title.csv",
        engine="python",
        on_bad_lines="skip"
    )

    ratings.rename(
        columns={"userId": "user_id", "movieId": "movie_id"},
        inplace=True
    )

    # CONTENT-BASED
    movies["genres"] = movies["genres"].fillna("")

    tfidf = TfidfVectorizer(stop_words="english")
    tfidf_matrix = tfidf.fit_transform(movies["genres"])

    similarity = cosine_similarity(tfidf_matrix)

    similarity_df = pd.DataFrame(
        similarity,
        index=movies["title"].values,
        columns=movies["title"].values
    )

    return movies, ratings, similarity_df


movies, ratings_base, similarity_df = load_data()

movie_list = sorted(movies["title"].dropna().unique())

# ---------------- USER INPUT ----------------
st.header("Step 1: Rate Movies")

m1 = st.selectbox("Movie 1", movie_list)
r1 = st.slider("Rating 1", 1.0, 5.0, 4.0)

m2 = st.selectbox("Movie 2", movie_list, index=1)
r2 = st.slider("Rating 2", 1.0, 5.0, 4.0)

m3 = st.selectbox("Movie 3", movie_list, index=2)
r3 = st.slider("Rating 3", 1.0, 5.0, 4.0)

user_input = [(m1, r1), (m2, r2), (m3, r3)]

# ---------------- RECOMMENDER ----------------
def recommend(user_input):

    watched = [m for m, _ in user_input]

    # ---------------- CONTENT BASED ----------------
    content_scores = {}

    for movie in watched:
        if movie not in similarity_df.columns:
            continue

        for title, score in similarity_df[movie].items():
            if title in watched:
                continue

            content_scores[title] = content_scores.get(title, 0) + score

    cb_df = pd.DataFrame(content_scores.items(), columns=["title", "cb_score"])

    # ---------------- SIMPLE COLLAB FILTER ----------------
    cf_scores = {}

    for movie in watched:
        user_ratings = [r for m, r in user_input if m == movie]
        if not user_ratings:
            continue

        for _, row in ratings_base.iterrows():
            if row["title"] in watched:
                continue
            cf_scores[row["title"]] = cf_scores.get(row["title"], 0) + user_ratings[0]

    cf_df = pd.DataFrame(cf_scores.items(), columns=["title", "cf_score"])

    # ---------------- HYBRID ----------------
    hybrid = pd.merge(cb_df, cf_df, on="title", how="inner")

    if hybrid.empty:
        return pd.DataFrame()

    # FAKE SVD SCORE (SAFE REPLACEMENT)
    hybrid["svd_score"] = 3.5

    hybrid["final_score"] = (
        0.5 * hybrid["cb_score"] +
        0.5 * hybrid["cf_score"]
    )

    hybrid = hybrid.sort_values("final_score", ascending=False).head(10)

    result = pd.merge(
        hybrid,
        movies[["title", "genres"]],
        on="title"
    )

    return result


# ---------------- OUTPUT ----------------
st.header("Step 2: Recommendations")

if st.button("Get Recommendations"):

    with st.spinner("Generating recommendations..."):
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
    
