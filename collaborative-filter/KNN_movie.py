import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors

class MovieKNNRecommender:
    def __init__(self, k=10):
        self.movies = pd.read_csv('../data/movies.csv')
        self.ratings = pd.read_csv('../data/ratings.csv')
        self.rating_matrix = self.ratings.pivot(index='movieId', columns='userId', values='rating').fillna(0)
        self.knn = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=k)
        self.knn.fit(self.rating_matrix)
    
    def get_movie_id(self, movie_title):
        movie_id = self.movies[self.movies['title'] == movie_title]['movieId']
        if len(movie_id) == 0:
            return None
        return movie_id.values[0]

    def get_similar_movies(self, movie_id, num=10):
        movie_index = self.rating_matrix.index.tolist().index(movie_id)
        distances, indices = self.knn.kneighbors([self.rating_matrix.iloc[movie_index]], n_neighbors=num+1)
        similar_movies = [self.rating_matrix.index[i] for i in indices.flatten()][1:]
        return similar_movies

    def recommend(self, movie_title, num=10):
        movie_id = self.get_movie_id(movie_title)
        if movie_id is None:
            return f"Movie '{movie_title}' not found."
        similar_movie_ids = self.get_similar_movies(movie_id, num)
        similar_movie_titles = self.movies[self.movies['movieId'].isin(similar_movie_ids)]['title']
        return similar_movie_titles.tolist()


recommender = MovieKNNRecommender()
movie_title = "Avengers: Infinity War - Part I (2018)"  
recommended_movies = recommender.recommend(movie_title, 10)
if isinstance(recommended_movies, list):
    for title in recommended_movies:
        print(title)
else:
    print(recommended_movies)
