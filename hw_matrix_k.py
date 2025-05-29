import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def calculate_error(data, user_matrix, item_matrix):
    total_error = 0
    for _, row in data.iterrows():
        user_idx = int(row['user']) - 1
        item_idx = int(row['item']) - 1
        predicted_rating = np.dot(user_matrix[user_idx], item_matrix[item_idx])
        total_error += (predicted_rating - row['rating'])**2
    return total_error / len(data)

def ALS(train_data, test_data, k, lambda1=0.01, lambda2=0.05, max_iterations=20, convergence_threshold=1e-4):
    num_users = train_data['user'].nunique()
    num_items = train_data['item'].nunique()

    
    user_matrix = np.random.rand(num_users, k)
    item_matrix = np.random.rand(num_items, k)

    prev_test_error = float('inf')

    for iteration in range(max_iterations):
        
        for i in range(num_users):
            relevant_items = train_data[train_data['user'] == i + 1]
            items = (relevant_items['item'] - 1).values 
            ratings = relevant_items['rating'].values
            item_matrix_T = item_matrix[items].T
            user_matrix[i] = np.linalg.solve(
                np.dot(item_matrix_T, item_matrix[items]) + lambda1 * np.eye(k),
                np.dot(item_matrix_T, ratings)
            )

       
        for j in range(num_items):
            relevant_users = train_data[train_data['item'] == j + 1]
            users = (relevant_users['user'] - 1).values 
            ratings = relevant_users['rating'].values
            user_matrix_T = user_matrix[users].T
            item_matrix[j] = np.linalg.solve(
                np.dot(user_matrix_T, user_matrix[users]) + lambda2 * np.eye(k),
                np.dot(user_matrix_T, ratings)
            )

       
        test_error = calculate_error(test_data, user_matrix, item_matrix)

      
        if abs(prev_test_error - test_error) < convergence_threshold:
            break

        prev_test_error = test_error

    return user_matrix, item_matrix, test_error


train_data = pd.read_csv('rate_train.csv', header=None, names=['user', 'item', 'rating'])
test_data = pd.read_csv('rate_test.csv', header=None, names=['user', 'item', 'rating'])


train_data['user'] = train_data['user'].astype(int)
train_data['item'] = train_data['item'].astype(int)

test_data['user'] = test_data['user'].astype(int)
test_data['item'] = test_data['item'].astype(int)


k_values = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]


testing_errors = {}


for k in k_values:
    
    user_matrix, item_matrix, test_error = ALS(train_data, test_data, k)
    testing_errors[k] = test_error


plt.plot(k_values, list(testing_errors.values()), marker='o')
plt.title('Converged Testing Error vs k')
plt.xlabel('k (Rank of Factorization)')
plt.ylabel('Converged Testing Error')
plt.show()
