import streamlit as st
import pickle
import requests
import matplotlib.pyplot as plt
import urllib.parse

# ================= PAGE CONFIG =================
st.set_page_config(
    page_title="Movie Recommender System",
    page_icon="🎬",
    layout="wide"
)

st.markdown("<h1 style='color:white;'>🎬 Movie Recommender System</h1>", unsafe_allow_html=True)
st.markdown(
    "<p style='color:gray;'>Genre-aware Content-Based Recommendation using NLP & Cosine Similarity</p>",
    unsafe_allow_html=True
)

# ================= LOAD MODELS =================
@st.cache_data
def load_models():
    movies = pickle.load(open("models/movie_list.pkl", "rb"))
    similarity = pickle.load(open("models/similarity.pkl", "rb"))
    return movies, similarity

movies, similarity = load_models()

# ================= POSTER FETCH =================
@st.cache_data(show_spinner=False)
def fetch_poster(movie_id):
    try:
        url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key=8265bd1679663a7ea12ac168da84d2e8"
        res = requests.get(url, timeout=5)
        if res.status_code == 200:
            data = res.json()
            if data.get("poster_path"):
                return "https://image.tmdb.org/t/p/w500" + data["poster_path"]
    except:
        pass

    # ALWAYS SHOW IMAGE
    return "https://via.placeholder.com/300x450.png?text=Poster+Not+Available"

# ================= TRAILER FETCH (100% WORKING) =================
@st.cache_data(show_spinner=False)
def fetch_trailer(movie_id, movie_title):
    """
    Priority:
    1. TMDB Trailer / Teaser / Clip
    2. YouTube search fallback (ALWAYS WORKS)
    """
    try:
        url = f"https://api.themoviedb.org/3/movie/{movie_id}/videos?api_key=8265bd1679663a7ea12ac168da84d2e8"
        res = requests.get(url, timeout=5)

        if res.status_code == 200:
            data = res.json()
            for v in data.get("results", []):
                if v.get("site") == "YouTube" and v.get("type") in ["Trailer", "Teaser", "Clip"]:
                    return f"https://www.youtube.com/watch?v={v['key']}"
    except:
        pass

    # 🔥 FALLBACK (GUARANTEED)
    query = urllib.parse.quote(f"{movie_title} official trailer")
    return f"https://www.youtube.com/results?search_query={query}"

# ================= RECOMMEND =================
def recommend(movie_name, selected_genre):
    index = movies[movies["title"] == movie_name].index[0]

    distances = list(enumerate(similarity[index]))
    distances = sorted(distances, key=lambda x: x[1], reverse=True)[1:15]

    names, posters, scores, trailers = [], [], [], []

    for i in distances:
        row = movies.iloc[i[0]]

        # GENRE FILTER
        if selected_genre != "All" and selected_genre not in row["genres"]:
            continue

        names.append(row["title"])
        posters.append(fetch_poster(row["id"]))
        scores.append(round(i[1] * 100, 2))
        trailers.append(fetch_trailer(row["id"], row["title"]))

        if len(names) == 5:
            break

    return names, posters, scores, trailers

# ================= UI =================
# GENRE DROPDOWN
all_genres = sorted({g for sub in movies["genres"] for g in sub})
selected_genre = st.selectbox("🎭 Select Genre", ["All"] + all_genres)

# MOVIE DROPDOWN
selected_movie = st.selectbox("🎥 Select a movie", movies["title"].values)

if st.button("🚀 Show Recommendations"):
    with st.spinner("Finding similar movies 🍿"):
        names, posters, scores, trailers = recommend(selected_movie, selected_genre)

    # ---------- MOVIE CARDS ----------
    st.markdown("<h3 style='color:white;'>🎯 Recommended Movies</h3>", unsafe_allow_html=True)
    cols = st.columns(5)

    for i in range(len(names)):
        with cols[i]:
            st.image(posters[i], use_container_width=True)
            st.markdown(
                f"<p style='color:white; font-weight:bold;'>{names[i]}</p>",
                unsafe_allow_html=True
            )
            st.markdown(
                f"<p style='color:#00ffc8;'>Similarity: {scores[i]}%</p>",
                unsafe_allow_html=True
            )
            # 🔥 ALWAYS CLICKABLE TRAILER BUTTON
            st.link_button("▶ Watch Trailer", trailers[i])

    # ---------- CRAZY GRAPH ----------
    st.markdown("<h3 style='color:white;'>📊 Similarity Comparison</h3>", unsafe_allow_html=True)

    fig, ax = plt.subplots(figsize=(9, 4))
    bars = ax.barh(names, scores, color="#00ffc8")

    fig.patch.set_facecolor("#0e1117")
    ax.set_facecolor("#0e1117")

    ax.set_xlabel("Similarity (%)", color="white")
    ax.set_xlim(0, 100)
    ax.invert_yaxis()
    ax.tick_params(axis="x", colors="white")
    ax.tick_params(axis="y", colors="white")

    for spine in ax.spines.values():
        spine.set_visible(False)

    for bar, score in zip(bars, scores):
        ax.text(
            score + 1,
            bar.get_y() + bar.get_height() / 2,
            f"{score}%",
            va="center",
            color="white",
            fontsize=11,
            fontweight="bold"
        )

    st.pyplot(fig)

    st.markdown(
        "<p style='color:gray;'>"
        "These movies are recommended because they share similar "
        "<b>genres, themes, cast, and keywords</b> with the selected movie. "
        "Cosine similarity measures how close their content is."
        "</p>",
        unsafe_allow_html=True
    )
