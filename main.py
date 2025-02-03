from fastapi import FastAPI, Depends
from contextlib import asynccontextmanager
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
import difflib

# Load data and initialize models at startup
movies_data = None
knn_model = None
feature_vectors = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global movies_data, knn_model, feature_vectors

    # Load data
    movies_data = pd.read_csv('Books.csv')

    # Data cleaning
    selected_features = ['Title','ISBN','Author','Publisher']
    for feature in selected_features:
        movies_data[feature] = movies_data[feature].fillna('')

    # Combine features
    combined_features = movies_data['Title']+' '+movies_data['ISBN']+' '+movies_data['Author']+' '+movies_data['Publisher']

    # Vectorization
    vectorizer = TfidfVectorizer()
    feature_vectors = vectorizer.fit_transform(combined_features)

    # Train KNN model
    knn_model = NearestNeighbors(metric='cosine', algorithm='brute')
    knn_model.fit(feature_vectors)

    yield  # FastAPI will use the initialized data and models here

    # Cleanup (if needed)
    movies_data = None
    knn_model = None
    feature_vectors = None

app = FastAPI(lifespan=lifespan)

@app.get("/{book_title}")
def read_root(book_title:str):
    global movies_data, knn_model, feature_vectors

    

    # Find the closest match
    list_of_all_titles = movies_data['Title'].tolist()
    find_close_match = difflib.get_close_matches(book_title, list_of_all_titles)
    close_match = find_close_match[0]

    # Get the index of the closest match
    index_of_the_movie = movies_data[movies_data.Title == close_match]['index'].values[0]

    # Find K nearest neighbors
    distances, indices = knn_model.kneighbors(feature_vectors[index_of_the_movie], n_neighbors=11)

    # Exclude the first one (input movie itself)
    recommended_movie_indices = indices.flatten()[1:]

    # Prepare the result
    result = []
    for index in recommended_movie_indices:
        title_from_index = movies_data[movies_data.index == index]['Title'].values[0]
        result.append(title_from_index)

    return result