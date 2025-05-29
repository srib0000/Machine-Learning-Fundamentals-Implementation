import numpy as np

data = np.loadtxt('diabetes.csv', delimiter=',')
[n, p] = np.shape(data)

# Set training number and testing number
num_train = int(0.75 * n)
num_test = int(0.25 * n)

# Split data into training set and testing set
sample_train = data[0:num_train, 0:-1]
sample_test = data[n - num_test:, 0:-1]
label_train = data[0:num_train, -1]
label_test = data[n - num_test:, -1]

def lda(X, y, reg_param=1e-2):
    n, p = X.shape
    classes = np.unique(y)
    means = []
    cov_matrices = []
    priors = []

    for c in classes:
        X_c = X[y == c]
        means.append(np.mean(X_c, axis=0))
        cov_matrices.append(np.cov(X_c, rowvar=False) + reg_param * np.identity(p))
        priors.append(len(X_c) / n)

    def predict(x):
        posteriors = []
        for i, c in enumerate(classes):
            cov_inv = np.linalg.inv(cov_matrices[i])
            diff = x - means[i]
            exponent = -0.5 * np.dot(np.dot(diff, cov_inv), diff)
            posterior = np.log(priors[i]) + exponent
            posteriors.append(posterior)
        return classes[np.argmax(posteriors)]

    return predict


def accuracy_score(y_true, y_pred):
    return 1 - np.mean(y_true == y_pred)  

lda_model = lda(sample_train, label_train, reg_param=1e-2)
label_train_pred_lda = [lda_model(x) for x in sample_train]
label_test_pred_lda = [lda_model(x) for x in sample_test]

# Calculate custom error score
error_train_lda = accuracy_score(label_train, label_train_pred_lda)
error_test_lda = accuracy_score(label_test, label_test_pred_lda)

# Print results
print("Training Error = %.4f" % error_train_lda)
print("Testing Error = %.4f" % error_test_lda)
