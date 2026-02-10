import pickle
import joblib

# load old pickle files
movies = pickle.load(open("models/movie_list.pkl", "rb"))
similarity = pickle.load(open("models/similarity.pkl", "rb"))

# save again using joblib (same file names)
joblib.dump(movies, "models/movie_list.pkl")
joblib.dump(similarity, "models/similarity.pkl")

print("✅ Conversion done successfully")
