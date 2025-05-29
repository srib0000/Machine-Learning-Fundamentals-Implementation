# Machine-Learning-Fundamentals-Implementation

This repository contains a comprehensive collection of 20+ assignments and projects completed as part of the Machine Learning Fundamentals course. The work spans supervised and unsupervised learning, model optimization, theoretical derivations, and practical evaluations using Python.

---

## 📂 Repository Structure

├── ridge_regression/ # Ridge regression + k-fold cross-validation
├── lasso_regression/ # Lasso via coordinate descent and sparsity analysis
├── logistic_regression/ # Gradient descent and Newton's method for MLE
├── linear_svm/ # Primal-dual form and KKT conditions
├── naive_bayes/ # Probabilistic classifier on binary dataset
├── decision_tree/ # Greedy tree construction and pruning
├── pca/ # Dimensionality reduction, reconstruction, projection
├── matrix_factorization/ # ALS-based recommender system
├── clustering_kmeans/ # PCA + K-means with DBI and Rand Index
├── knn/ # Compressed kNN and error vs. τ and k
├── lda/ # Linear Discriminant Analysis (LDA) implementation
├── akrr/ # Accelerated Kernel Ridge Regression
├── ann/ # Simple neural net with backpropagation
├── ensemble_bagging/ # Bagging with bootstrap sampling
├── map_vs_rwls/ # Equivalence between MAP and RWLS
├── weighted_least_squares/ # WLS with subgroup-specific weighting
├── bonus/ # Kernel validity proof, bias-variance decomposition
├── datasets/ # CSV data files: crime, diabetes, rating data
├── plots/ # All result plots (PNG)
├── midterm/ # Final integrated testing scripts


---

## ✅ Key Topics Covered

- 📈 **Regression**: Ridge, Lasso, Logistic, WLS
- 🧠 **Classification**: LDA, Naive Bayes, k-NN, SVM
- 🌲 **Tree-Based Methods**: Decision Trees, Bagging
- 🔍 **Dimensionality Reduction**: PCA
- 🤝 **Matrix Factorization**: ALS-based collaborative filtering
- 🔢 **Optimization**: GD, Newton’s Method, CD, ALS
- 📊 **Evaluation**: CV, DBI, Rand Index, Error vs. λ, k, τ

---

## 🛠️ How to Run

Each folder contains:
- Python scripts (`.py`)
- Required datasets (`.csv`)
- Output plots (`.png`)
- Assignment writeups (`.pdf`)

To reproduce results:
```bash
python hw_lasso_ercd.py    # Example: run coordinate descent for Lasso
python hw_rr_cv.py         # Run k-fold CV for Ridge Regression

pip install numpy pandas matplotlib scikit-learn


