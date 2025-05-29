import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def calculate_error(data, user_matrix, movie_matrix):
    total_error = 0
    for _, row in data.iterrows():
        user_idx = int(row['user']) - 1
        movie_idx = int(row['movie']) - 1
        predicted_rating = np.dot(user_matrix[user_idx], movie_matrix[movie_idx])
        total_error += (predicted_rating - row['rating'])**2
    return total_error / len(data)

def ALS(train_data, test_data, k, lambda1=0.01, lambda2=0.05, max_iterations=20, convergence_threshold=1e-4):
    num_users = train_data['user'].nunique()
    num_movies = train_data['movie'].nunique()

   
    user_matrix = np.random.rand(num_users, k)
    movie_matrix = np.random.rand(num_movies, k)

    prev_test_error = float('inf')

    testing_errors = []

    for iteration in range(max_iterations):
       
        for i in range(num_users):
            relevant_movies = train_data[train_data['user'] == i + 1]
            movies = (relevant_movies['movie'] - 1).values  # Adjust index to start from 0
            ratings = relevant_movies['rating'].values
            movie_matrix_T = movie_matrix[movies].T
            user_matrix[i] = np.linalg.solve(
                np.dot(movie_matrix_T, movie_matrix[movies]) + lambda1 * np.eye(k),
                np.dot(movie_matrix_T, ratings)
            )

    
        for j in range(num_movies):
            relevant_users = train_data[train_data['movie'] == j + 1]
            users = (relevant_users['user'] - 1).values  # Adjust index to start from 0
            ratings = relevant_users['rating'].values
            user_matrix_T = user_matrix[users].T
            movie_matrix[j] = np.linalg.solve(
                np.dot(user_matrix_T, user_matrix[users]) + lambda2 * np.eye(k),
                np.dot(user_matrix_T, ratings)
            )

       
        test_error = calculate_error(test_data, user_matrix, movie_matrix)

        testing_errors.append(test_error)

    return user_matrix, movie_matrix, testing_errors


train_data = pd.read_csv('rate_train.csv', header=None, names=['user', 'movie', 'rating'])
test_data = pd.read_csv('rate_test.csv', header=None, names=['user', 'movie', 'rating'])


train_data['user'] = train_data['user'].astype(int)
train_data['movie'] = train_data['movie'].astype(int)

test_data['user'] = test_data['user'].astype(int)
test_data['movie'] = test_data['movie'].astype(int)


k = 10
lambda1 = 0.01 
lambda2 = 0.05  
max_iterations = 20 


user_matrix, movie_matrix, testing_errors = ALS(train_data, test_data, k, lambda1, lambda2, max_iterations)


plt.plot(range(1, len(testing_errors) + 1), testing_errors, marker='o')
plt.title('Testing Error vs Number of Updates')
plt.xlabel('Number of Updates')
plt.ylabel('Testing Error')
plt.show()
