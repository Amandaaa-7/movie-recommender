import streamlit as st
import pandas as pd
import joblib
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
# LOAD DATA + MODEL
# =========================
movies = pd.read_csv("clean_content.csv")
ratings = pd.read_csv("ratings_title.csv", engine="python", on_bad_lines="skip")

svd_model = joblib.load("svd_model.pkl")

# =========================
# CONTENT-BASED MODEL
# =========================
movies["genres"] = movies["genres"].fillna("")

tfidf = TfidfVectorizer(stop_words="english")
tfidf_matrix = tfidf.fit_transform(movies["genres"])
similarity = cosine_similarity(tfidf_matrix)

sim_df = pd.DataFrame(similarity,
                      index=movies["title"],
                      columns=movies["title"])

# =========================
# UI TITLE
# =========================
st.title("🎬 Hybrid Movie Recommendation System")

language = st.selectbox("🌍 Language", ["English", "Arabic", "French"])

TEXT = {
    "English": {"btn": "Get Recommendations", "empty": "No recommendations"},
    "Arabic": {"btn": "عرض التوصيات", "empty": "لا توجد نتائج"},
    "French": {"btn": "Voir recommandations", "empty": "Aucun résultat"}
}

# =========================
# USER INPUT
# =========================
movie_list = movies["title"].dropna().unique()

m1 = st.selectbox("Movie 1", movie_list)
m2 = st.selectbox("Movie 2", movie_list, index=1)
m3 = st.selectbox("Movie 3", movie_list, index=2)

u1 = st.slider("Rating 1", 1, 5, 4)
u2 = st.slider("Rating 2", 1, 5, 4)
u3 = st.slider("Rating 3", 1, 5, 4)

user_input = [(m1, u1), (m2, u2), (m3, u3)]

# =========================
# HELPER FUNCTIONS
# =========================

def normalize(x):
    return (x - 1) / 4  # scale 1–5 → 0–1


def get_cf_score(movie_title):
    try:
        movie_id = ratings[ratings["title"] == movie_title]["movie_id"].values[0]
        pred = svd_model.predict(1, movie_id)  # user_id fixed for demo
        return pred.est
    except:
        return 3.0


def recommend(user_input):

    watched = [m for m, _ in user_input]

    # ---------------- CONTENT ----------------
    content_scores = {}

    for movie in watched:
        if movie not in sim_df.columns:
            continue

        for title, score in sim_df[movie].items():
            if title in watched:
                continue
            content_scores[title] = content_scores.get(title, 0) + score

    # ---------------- CF (SVD) ----------------
    cf_scores = {}

    for movie, _ in user_input:
        for title in movies["title"]:
            if title in watched:
                continue
            cf_scores[title] = get_cf_score(title)

    if not content_scores or not cf_scores:
        return pd.DataFrame()

    # ---------------- MERGE ----------------
    df_cb = pd.DataFrame(content_scores.items(), columns=["title", "cb"])
    df_cf = pd.DataFrame(cf_scores.items(), columns=["title", "cf"])

    hybrid = pd.merge(df_cb, df_cf, on="title")

    # ---------------- NORMALIZE ----------------
    hybrid["cb"] = hybrid["cb"] / hybrid["cb"].max()
    hybrid["cf"] = hybrid["cf"].apply(normalize)

    # ---------------- FINAL SCORE ----------------
    hybrid["final_score"] = (
        0.5 * hybrid["cb"] +
        0.5 * hybrid["cf"]
    ) * 5  # scale back to 1–5

    hybrid["final_score"] = hybrid["final_score"].clip(1, 5)

    hybrid = hybrid.sort_values("final_score", ascending=False).head(10)

    return pd.merge(hybrid, movies[["title", "genres"]], on="title")


# =========================
# OUTPUT
# =========================
if st.button(TEXT[language]["btn"]):

    recs = recommend(user_input)

    if recs.empty:
        st.warning(TEXT[language]["empty"])
    else:
        for _, row in recs.iterrows():
            st.markdown(f"""
            **🎥 {row['title']}**  
            🎭 {row['genres']}  
            ⭐ Score: {round(row['final_score'], 2)}
            ---
            """)
