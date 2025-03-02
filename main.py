from fastapi import FastAPI, Depends
from contextlib import asynccontextmanager
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
import difflib



app = FastAPI()

@app.get("/")
def read_root(bookname:str):
   movies_data = pd.read_csv('bbe.csv')
   movie_name = bookname
   # Ensure index column exists or reset index
   movies_data = movies_data.reset_index()
   print("TITLE",bookname)
# Data cleaning
   selected_features = ['title', 'author', 'genres', 'series', 'likedPercent']
   for feature in selected_features:
     movies_data[feature] = movies_data[feature].fillna('')

# Convert likedPercent to string (if it's numeric)
   movies_data['likedPercent'] = movies_data['likedPercent'].astype(str)

# Create combined features
   combined_features = movies_data['title'] + ' ' + movies_data['author'] +  movies_data['genres']  + movies_data['series'] + ' ' + movies_data['likedPercent']

# Vectorization
   vectorizer = TfidfVectorizer()
   feature_vectors = vectorizer.fit_transform(combined_features)

# Setting up KNN with cosine distance
   knn_model = NearestNeighbors(metric='cosine', algorithm='brute')
   knn_model.fit(feature_vectors)

# Receiving the input
  

# Finding the matched movie title
   list_of_all_titles = movies_data['title'].tolist()
   find_close_match = difflib.get_close_matches(movie_name, list_of_all_titles)

   if not find_close_match:
     print("No close match found. Try another movie.")
   else:
    close_match = find_close_match[0]

    # Get movie index
    index_of_the_movie = movies_data[movies_data['title'] == close_match].index[0]

    # Find K nearest neighbors
    distances, indices = knn_model.kneighbors(feature_vectors[index_of_the_movie], n_neighbors=11)

    # Exclude the first one as it is the input movie itself
    recommended_movie_indices = indices.flatten()[1:]

    # Display movie titles based on knn model
    print('\nMovies suggested for you:\n')
    title_from_index=[]
    for i, index in enumerate(recommended_movie_indices, start=1):
        title_from_index.append({"title":movies_data.iloc[index]['title'],"cover":movies_data.iloc[index]['coverImg']})
        
    
   return title_from_index