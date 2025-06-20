import os
import pickle
import pandas as pd
import streamlit as st
import requests
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import ast
import nltk
from nltk.stem.porter import PorterStemmer

# Download tokenizer if not already present
nltk.download('punkt')

# Setup
st.set_page_config(page_title="Movie Recommender System")
st.header('🎬 Movie Recommender System')

# Stemmer
ps = PorterStemmer()

def stem(text):
    return " ".join([ps.stem(word) for word in text.split()])

# === AUTO BUILD SECTION ===
if not os.path.exists("movie_list.pkl") or not os.path.exists("similarity.pkl"):
    # Load raw data
    movies_raw = pd.read_csv("tmdb_5000_movies.csv")
    credits = pd.read_csv("tmdb_5000_credits.csv")

    movies_raw = movies_raw.merge(credits, on='title')

    # Select needed columns
    movies_raw = movies_raw[['movie_id', 'title', 'overview', 'genres', 'keywords', 'cast', 'crew']]
    movies_raw.dropna(inplace=True)

    def convert(obj):
        L = []
        for i in ast.literal_eval(obj):
            L.append(i['name'])
        return L

    def convert_cast(obj):
        L = []
        count = 0
        for i in ast.literal_eval(obj):
            if count < 3:
                L.append(i['name'])
                count += 1
            else:
                break
        return L

    def fetch_director(obj):
        for i in ast.literal_eval(obj):
            if i['job'] == 'Director':
                return [i['name']]
        return []

    # Apply conversions
    movies_raw['genres'] = movies_raw['genres'].apply(convert)
    movies_raw['keywords'] = movies_raw['keywords'].apply(convert)
    movies_raw['cast'] = movies_raw['cast'].apply(convert_cast)
    movies_raw['crew'] = movies_raw['crew'].apply(fetch_director)
    movies_raw['overview'] = movies_raw['overview'].apply(lambda x: x.split())

    movies_raw['tags'] = movies_raw['overview'] + movies_raw['genres'] + movies_raw['keywords'] + movies_raw['cast'] + movies_raw['crew']

    new_df = movies_raw[['movie_id', 'title', 'tags']]
    new_df['tags'] = new_df['tags'].apply(lambda x: " ".join(x))
    new_df['tags'] = new_df['tags'].apply(lambda x: x.lower())
    new_df['tags'] = new_df['tags'].apply(stem)

    # Vectorize
    cv = CountVectorizer(max_features=5000, stop_words='english')
    vectors = cv.fit_transform(new_df['tags']).toarray()

    similarity = cosine_similarity(vectors)

    # Save
    pickle.dump(new_df, open('movie_list.pkl', 'wb'))
    pickle.dump(similarity, open('similarity.pkl', 'wb'))
else:
    new_df = pickle.load(open('movie_list.pkl', 'rb'))
    similarity = pickle.load(open('similarity.pkl', 'rb'))
# === END OF BUILD ===


# TMDB Poster Fetcher
def fetch_poster(movie_id):
    try:
        url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key=8265bd1679663a7ea12ac168da84d2e8&language=en-US"
        data = requests.get(url)
        data = data.json()
        poster_path = data.get('poster_path', '')
        if poster_path:
            full_path = "https://image.tmdb.org/t/p/w500/" + poster_path
            return full_path
    except:
        return "https://via.placeholder.com/500x750?text=Poster+Not+Available"
    return "https://via.placeholder.com/500x750?text=Poster+Not+Available"

# Recommendation Logic
def recommend(movie):
    index = new_df[new_df['title'] == movie].index[0]
    distances = sorted(list(enumerate(similarity[index])), reverse=True, key=lambda x: x[1])
    recommended_movie_names = []
    recommended_movie_posters = []
    for i in distances[1:6]:
        movie_id = new_df.iloc[i[0]].movie_id
        recommended_movie_names.append(new_df.iloc[i[0]].title)
        recommended_movie_posters.append(fetch_poster(movie_id))
    return recommended_movie_names, recommended_movie_posters


# App UI
movie_list = new_df['title'].values
selected_movie = st.selectbox("🎥 Type or select a movie", movie_list)

if st.button('Show Recommendation'):
    names, posters = recommend(selected_movie)
    cols = st.columns(5)
    for i in range(5):
        with cols[i]:
            st.text(names[i])
            st.image(posters[i])
