import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Movie Recommendation System", layout="centered")

st.title("🎬 Movie Recommendation System")
st.write("Content-Based Recommendation using Cosine Similarity")

# Load dataset
movies = pd.read_csv("movies.csv")
movies['genres'] = movies['genres'].str.replace('|', ' ', regex=False)
movies['title_lower'] = movies['title'].str.lower()

# Vectorization
cv = CountVectorizer(stop_words='english')
genre_matrix = cv.fit_transform(movies['genres'])

# Similarity matrix
similarity = cosine_similarity(genre_matrix)

def recommend(movie_title):
    movie_title = movie_title.lower()
    matches = movies[movies['title_lower'].str.contains(movie_title)]

    if matches.empty:
        return []

    idx = matches.index[0]
    scores = list(enumerate(similarity[idx]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)[1:6]

    return [movies.iloc[i[0]].title for i in scores]


movie_input = st.text_input("Enter a movie name")

if st.button("Recommend"):
    if movie_input:
        recommendations = recommend(movie_input)

        if recommendations:
            st.subheader("Recommended Movies:")
            for movie in recommendations:
                st.write("🎥", movie)
        else:
            st.warning("Movie not found. Try another title.")
