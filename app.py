import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="Movie Recommender",
    page_icon="🎬",
    layout="centered"
)

# =========================
# CUSTOM CSS
# =========================
st.markdown("""
<style>

.main {
    background-color: #f5f7fa;
}

.stButton>button {
    width: 100%;
    background-color: #DC2626;
    color: white;
    border-radius: 12px;
    height: 50px;
    font-size: 18px;
    font-weight: bold;
    border: none;
}

.stButton>button:hover {
    background-color: #B91C1C;
    color: white;
}

.title {
    text-align: center;
    font-size: 42px;
    font-weight: bold;
    color: #991B1B;
}

.subtitle {
    text-align: center;
    color: gray;
    margin-bottom: 30px;
}

.movie-card {
    background-color: white;
    padding: 15px;
    border-radius: 12px;
    margin-bottom: 15px;
    border-left: 6px solid #DC2626;
    box-shadow: 0px 2px 6px rgba(0,0,0,0.1);
}

</style>
""", unsafe_allow_html=True)

# =========================
# TITLE
# =========================
st.markdown(
    '<p class="title">🎬 Hybrid Movie Recommendation System</p>',
    unsafe_allow_html=True
)

st.markdown(
    '<p class="subtitle">Get personalized movie recommendations using AI</p>',
    unsafe_allow_html=True
)

# =========================
# LANGUAGE
# =========================
language = st.selectbox(
    "🌍 Language",
    ["English", "العربية", "Français"]
)

TEXT = {
    "English": {
        "button": "Get Recommendations",
        "recommend": "Recommended Movies",
        "warning": "Not enough data to generate recommendations.",
        "genres": "Genres",
        "score": "Score"
    },

    "العربية": {
        "button": "عرض التوصيات",
        "recommend": "الأفلام المقترحة",
        "warning": "لا توجد بيانات كافية لإنشاء التوصيات.",
        "genres": "النوع",
        "score": "التقييم"
    },

    "Français": {
        "button": "Obtenir des recommandations",
        "recommend": "Films recommandés",
        "warning": "Pas assez de données pour générer des recommandations.",
        "genres": "Genres",
        "score": "Score"
    }
}

# =========================
# LOAD DATA
# =========================
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

movie_list = sorted(
    movies["title"].dropna().unique()
)

# =========================
# USER INPUT
# =========================
st.subheader("🎥 Select Movies")

m1 = st.selectbox("Movie 1", movie_list)
r1 = st.slider("Rating 1", 1.0, 5.0, 4.0)

m2 = st.selectbox("Movie 2", movie_list, index=1)
r2 = st.slider("Rating 2", 1.0, 5.0, 4.0)

m3 = st.selectbox("Movie 3", movie_list, index=2)
r3 = st.slider("Rating 3", 1.0, 5.0, 4.0)

user_input = [
    (m1, r1),
    (m2, r2),
    (m3, r3)
]

# =========================
# RECOMMENDER
# =========================
def recommend(user_input):

    watched = [m for m, _ in user_input]

    content_scores = {}

    for movie in watched:

        if movie not in similarity_df.columns:
            continue

        for title, score in similarity_df[movie].items():

            if title in watched:
                continue

            content_scores[title] = (
                content_scores.get(title, 0) + score
            )

    cb_df = pd.DataFrame(
        content_scores.items(),
        columns=["title", "cb_score"]
    )

    cf_scores = {}

    for movie in watched:

        user_ratings = [
            r for m, r in user_input if m == movie
        ]

        if not user_ratings:
            continue

        for _, row in ratings_base.iterrows():

            if row["title"] in watched:
                continue

            cf_scores[row["title"]] = (
                cf_scores.get(row["title"], 0)
                + user_ratings[0]
            )

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
        0.5 * hybrid["cb_score"] +
        0.5 * hybrid["cf_score"]
    )

    hybrid = hybrid.sort_values(
        "final_score",
        ascending=False
    ).head(10)

    result = pd.merge(
        hybrid,
        movies[["title", "genres"]],
        on="title"
    )

    return result

# =========================
# BUTTON
# =========================
if st.button(TEXT[language]["button"]):

    with st.spinner("Loading Recommendations..."):

        recs = recommend(user_input)

    if recs.empty:

        st.warning(TEXT[language]["warning"])

    else:

        st.subheader(f"🍿 {TEXT[language]['recommend']}")

        for _, row in recs.iterrows():

            st.markdown(
                f"""
                <div class="movie-card">
                    <h4>🎬 {row['title']}</h4>
                    <p><b>{TEXT[language]['genres']}:</b> {row['genres']}</p>
                    <p><b>{TEXT[language]['score']}:</b> {round(row['final_score'], 2)}</p>
                </div>
                """,
                unsafe_allow_html=True
            )
