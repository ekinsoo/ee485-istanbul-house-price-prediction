import numpy as np

class CustomRidgeRegression:
    def __init__(self, alpha=1.0):
        self.alpha = alpha
        self.coef_ = None
        self.intercept_ = None

    def fit(self, X, y):
        #bias term eklenecek 
        X_b = np.c_[np.ones((X.shape[0], 1)), X]
        
        #identity matrix for regularization
        I = np.eye(X_b.shape[1])
        I[0, 0] = 0  # Do not regularize the bias term
        
        # Closed form solution: theta = (X^T * X + alpha * I)^-1 * X^T * y
        # Using pseudo-inverse for stability
        A = X_b.T.dot(X_b) + self.alpha * I
        b = X_b.T.dot(y)
        
        try:
            theta = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            theta = np.linalg.pinv(A).dot(b)
            
        self.intercept_ = theta[0]
        self.coef_ = theta[1:]
        return self

    def predict(self, X):
        return X.dot(self.coef_) + self.intercept_
# en iyi ridge modelini bulabilmek icin degisik alpha degerleri cross validation ile
def train_ridge(X_train, y_train, alphas=[0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]):
    print(f"Training Custom Ridge Regression with Cross-Validation...")
    
    best_alpha = None
    best_score = -np.inf
    
    # 5 fold cross-validation
    k = 5
    fold_size = len(X_train) // k
    indices = np.arange(len(X_train))
    np.random.shuffle(indices)
    
    for alpha in alphas:
        scores = []
        for i in range(k):
            val_idx = indices[i*fold_size : (i+1)*fold_size]
            train_idx = np.concatenate([indices[:i*fold_size], indices[(i+1)*fold_size:]])
            X_fold_train, X_fold_val = X_train[train_idx], X_train[val_idx]
            y_fold_train, y_fold_val = y_train[train_idx], y_train[val_idx]
            
            model = CustomRidgeRegression(alpha=alpha)
            model.fit(X_fold_train, y_fold_train)
            y_pred = model.predict(X_fold_val)
            
            #r2 score calculation part, mse too much since the prices are much
            ss_res = np.sum((y_fold_val - y_pred) ** 2)
            ss_tot = np.sum((y_fold_val - np.mean(y_fold_val)) ** 2)
            r2 = 1 - (ss_res / ss_tot)
            scores.append(r2)
            
        avg_score = np.mean(scores)
        #alpha degerini ciktiya gore optimize etme algoritmasi
        if avg_score > best_score:
            best_score = avg_score
            best_alpha = alpha
            
    print(f"Best alpha found: {best_alpha}")
    final_model = CustomRidgeRegression(alpha=best_alpha)
    final_model.fit(X_train, y_train)
    return final_model

def evaluate_ridge(model, X_test, y_test):
    y_pred = model.predict(X_test)
    
    # mean squared error calculation
    mse = np.mean((y_test - y_pred) ** 2)  
    ss_res = np.sum((y_test - y_pred) ** 2)
    ss_tot = np.sum((y_test - np.mean(y_test)) ** 2)
    r2 = 1 - (ss_res / ss_tot)
    
    return mse, r2, y_pred
