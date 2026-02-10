import pandas as pd
import pickle
import ast
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load data
movies = pd.read_csv("data/tmdb_5000_movies.csv")
credits = pd.read_csv("data/tmdb_5000_credits.csv")

movies = movies.merge(credits, on="title")

# Helpers
def convert(text):
    return [i["name"] for i in ast.literal_eval(text)]

def get_top_cast(text):
    return [i["name"] for i in ast.literal_eval(text)[:3]]

def get_director(text):
    for i in ast.literal_eval(text):
        if i["job"] == "Director":
            return i["name"]
    return ""

# Process features
movies["genres"] = movies["genres"].apply(convert)
movies["keywords"] = movies["keywords"].apply(convert)
movies["cast"] = movies["cast"].apply(get_top_cast)
movies["director"] = movies["crew"].apply(get_director)
movies["overview"] = movies["overview"].fillna("")

movies["tags"] = (
    movies["overview"]
    + " " + movies["genres"].apply(lambda x: " ".join(x))
    + " " + movies["keywords"].apply(lambda x: " ".join(x))
    + " " + movies["cast"].apply(lambda x: " ".join(x))
    + " " + movies["director"]
)

movies["tags"] = movies["tags"].str.lower()

# Vectorization
cv = CountVectorizer(max_features=5000, stop_words="english")
vectors = cv.fit_transform(movies["tags"]).toarray()
similarity = cosine_similarity(vectors)

# Save required columns (IMPORTANT)
final_movies = movies[["id", "title", "genres"]]

pickle.dump(final_movies, open("models/movie_list.pkl", "wb"))
pickle.dump(similarity, open("models/similarity.pkl", "wb"))

print("✅ Models regenerated with GENRE support")
