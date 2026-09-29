import numpy as np

class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature      # split feature index
        self.threshold = threshold  # split threshold
        self.left = left           # left child node
        self.right = right         # right child node
        self.value = value         # leaf node value (mean of y)

class CustomDecisionTree:
    def __init__(self, max_depth=10, min_samples_split=10, min_samples_leaf=5):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.root = None
    
    def _mse(self, y):
        # mean squared error hesapla
        if len(y) == 0:
            return 0
        mean = np.mean(y)
        return np.mean((y - mean) ** 2)
    
    def _best_split(self, X, y):
        # en iyi split noktasini bul (greedy)
        m, n = X.shape
        if m <= self.min_samples_split:
            return None, None
        
        best_mse = float('inf')
        best_feature = None
        best_threshold = None
        
        parent_mse = self._mse(y)
        
        # her feature icin
        for feature_idx in range(n):
            # unique degerler al ve sort et
            thresholds = np.unique(X[:, feature_idx])
            
            # eger cok fazla unique deger varsa sample al (hiz icin)
            if len(thresholds) > 10:
                thresholds = np.percentile(X[:, feature_idx], np.linspace(10, 90, 10))
            
            # her threshold icin
            for threshold in thresholds:
                # split yap
                left_mask = X[:, feature_idx] <= threshold
                right_mask = ~left_mask
                
                # min samples kontrolu
                if np.sum(left_mask) < self.min_samples_leaf or np.sum(right_mask) < self.min_samples_leaf:
                    continue
                
                # MSE hesapla
                left_mse = self._mse(y[left_mask])
                right_mse = self._mse(y[right_mask])
                
                # weighted MSE
                n_left = np.sum(left_mask)
                n_right = np.sum(right_mask)
                weighted_mse = (n_left * left_mse + n_right * right_mse) / m
                
                # en iyi split'i bul
                if weighted_mse < best_mse:
                    best_mse = weighted_mse
                    best_feature = feature_idx
                    best_threshold = threshold
        
        return best_feature, best_threshold
    
    def _build_tree(self, X, y, depth=0):
        # recursive olarak tree olustur
        n_samples = X.shape[0]
        
        # stopping conditions
        if depth >= self.max_depth or n_samples < self.min_samples_split:
            leaf_value = np.mean(y)
            return Node(value=leaf_value)
        
        # en iyi split'i bul
        feature, threshold = self._best_split(X, y)
        
        # eger split bulunamazsa leaf node olustur
        if feature is None:
            leaf_value = np.mean(y)
            return Node(value=leaf_value)
        
        # split yap
        left_mask = X[:, feature] <= threshold
        right_mask = ~left_mask
        
        # recursive olarak child node'lari olustur
        left_child = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        right_child = self._build_tree(X[right_mask], y[right_mask], depth + 1)
        
        return Node(feature=feature, threshold=threshold, left=left_child, right=right_child)
    
    def fit(self, X, y):
        # tree'yi train et
        print("Building Decision Tree...")
        self.root = self._build_tree(X, y)
        return self
    
    def _predict_sample(self, x, node):
        # tek bir sample icin prediction
        if node.value is not None:
            return node.value
        
        if x[node.feature] <= node.threshold:
            return self._predict_sample(x, node.left)
        else:
            return self._predict_sample(x, node.right)
    
    def predict(self, X):
        # tum samples icin prediction
        return np.array([self._predict_sample(x, self.root) for x in X])

def train_decision_tree(X_train, y_train, max_depth=10):
    print(f"Training Custom Decision Tree (max_depth={max_depth})...")
    
    model = CustomDecisionTree(
        max_depth=max_depth,
        min_samples_split=10,
        min_samples_leaf=5
    )
    
    model.fit(X_train, y_train)
    return model

def evaluate_decision_tree(model, X_test, y_test):
    # test seti uzerinde degerlendir
    y_pred = model.predict(X_test)
    
    mse = np.mean((y_test - y_pred) ** 2)
    ss_res = np.sum((y_test - y_pred) ** 2)
    ss_tot = np.sum((y_test - np.mean(y_test)) ** 2)
    r2 = 1 - (ss_res / ss_tot)
    
    return mse, r2, y_pred
