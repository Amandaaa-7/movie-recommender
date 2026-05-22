import streamlit as st
import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(
    page_title="Movie Recommendation System",
    page_icon="🎬",
    layout="centered"
)

st.markdown("""
<style>

.main {
    background-color: #f5f7fa;
}

.title {
    text-align: center;
    font-size: 42px;
    font-weight: bold;
    color: #7C3AED;
}

.subtitle {
    text-align: center;
    color: gray;
    margin-bottom: 30px;
}

.stButton>button {
    width: 100%;
    background-color: #7C3AED;
    color: white;
    border-radius: 12px;
    height: 50px;
    font-size: 18px;
    font-weight: bold;
    border: none;
}

.stButton>button:hover {
    background-color: #6D28D9;
    color: white;
}

.result-box {
    background-color: #EDE9FE;
    padding: 15px;
    border-radius: 12px;
    color: #4C1D95;
    font-size: 18px;
    font-weight: bold;
}

</style>
""", unsafe_allow_html=True)

st.markdown(
    '<p class="title">🎬 Hybrid Movie Recommendation System</p>',
    unsafe_allow_html=True
)

st.markdown(
    '<p class="subtitle">Get personalized movie recommendations using Hybrid Recommendation</p>',
    unsafe_allow_html=True
)

language = st.selectbox(
    "🌍 Language",
    ["English", "العربية", "Français"]
)

TEXT = {
    "English": {
        "button": "Get Recommendations",
        "header": "Recommended Movies",
        "empty": "No recommendations found"
    },
    "العربية": {
        "button": "عرض التوصيات",
        "header": "الأفلام المقترحة",
        "empty": "لا توجد توصيات"
    },
    "Français": {
        "button": "Obtenir des recommandations",
        "header": "Films recommandés",
        "empty": "Aucune recommandation trouvée"
    }
}

@st.cache_data
def load_data():

    movies = pd.read_csv("clean_content.csv")

    ratings = pd.read_csv(
        "ratings_title.csv",
        engine="python",
        on_bad_lines="skip"
    )

    ratings.rename(
        columns={
            "userId": "user_id",
            "movieId": "movie_id"
        },
        inplace=True
    )

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

st.subheader("🎥 Select Movies")

movie1 = st.selectbox("Movie 1", movie_list)
rating1 = st.slider("Rating 1", 1.0, 5.0, 4.0)

movie2 = st.selectbox("Movie 2", movie_list, index=1)
rating2 = st.slider("Rating 2", 1.0, 5.0, 4.0)

movie3 = st.selectbox("Movie 3", movie_list, index=2)
rating3 = st.slider("Rating 3", 1.0, 5.0, 4.0)

user_input = [
    (movie1, rating1),
    (movie2, rating2),
    (movie3, rating3)
]

def recommend(user_input):

    watched = [m for m, r in user_input]

    content_scores = {}

    for movie in watched:

        if movie not in similarity_df.columns:
            continue

        similar_movies = similarity_df[movie].sort_values(ascending=False)

        for title, score in similar_movies.items():

            if title in watched:
                continue

            content_scores[title] = (
                content_scores.get(title, 0) + score
            )

    if len(content_scores) == 0:
        return pd.DataFrame()

    cb_df = pd.DataFrame(
        content_scores.items(),
        columns=["title", "content_score"]
    )

    cf_scores = {}

    for movie, rating in user_input:

        related = ratings_base[
            ratings_base["title"] == movie
        ]

        if related.empty:
            continue

        users = related["user_id"].unique()

        similar_user_movies = ratings_base[
            ratings_base["user_id"].isin(users)
        ]

        for title in similar_user_movies["title"]:

            if title in watched:
                continue

            cf_scores[title] = (
                cf_scores.get(title, 0) + rating
            )

    if len(cf_scores) == 0:
        return pd.DataFrame()

    cf_df = pd.DataFrame(
        cf_scores.items(),
        columns=["title", "cf_score"]
    )

    hybrid = pd.merge(
        cb_df,
        cf_df,
        on="title",
        how="inner"
    )

    if hybrid.empty:
        return pd.DataFrame()

    hybrid["final_score"] = (
        0.5 * hybrid["content_score"] +
        0.5 * hybrid["cf_score"]
    )

    hybrid = hybrid.sort_values(
        "final_score",
        ascending=False
    ).head(10)

    result = pd.merge(
        hybrid,
        movies[["title", "genres"]],
        on="title",
        how="left"
    )

    return result

if st.button(TEXT[language]["button"]):

    try:

        recommendations = recommend(user_input)

        st.subheader(f"🎬 {TEXT[language]['header']}")

        if recommendations.empty:

            st.warning(TEXT[language]["empty"])

        else:

            for i, row in recommendations.iterrows():

                st.markdown(
                    f"""
                    <div class="result-box">
                    🎥 {row['title']} <br>
                    🎭 {row['genres']} <br>
                    ⭐ Score: {round(row['final_score'], 2)}
                    </div>
                    <br>
                    """,
                    unsafe_allow_html=True
                )

    except Exception as e:

        st.error(f"Error: {e}")
