import numpy as np

class CustomNeuralNetwork:
    def __init__(self, hidden_layers=[128, 64, 32], learning_rate=0.005, epochs=300, batch_size=64, random_state=42, momentum=0.9, l2_lambda=0.001, dropout_rate=0.2):
        self.hidden_layers = hidden_layers
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.random_state = random_state
        self.momentum = momentum
        self.l2_lambda = l2_lambda
        self.dropout_rate = dropout_rate
        self.weights = []
        self.biases = []
        self.loss_history = []
        self.y_mean = 0
        self.y_std = 1
        
        # Momentum velocities
        self.v_w = []
        self.v_b = []
        
    def _initialize_weights(self, input_dim):
        # xavier initialization ile agirlik baslatma
        np.random.seed(self.random_state)
        layers = [input_dim] + self.hidden_layers + [1]
        
        self.weights = []
        self.biases = []
        self.v_w = []
        self.v_b = []
        
        for i in range(len(layers) - 1):
            # Xavier initialization
            limit = np.sqrt(6 / (layers[i] + layers[i+1]))
            w = np.random.uniform(-limit, limit, (layers[i], layers[i+1]))
            b = np.zeros((1, layers[i+1]))
            self.weights.append(w)
            self.biases.append(b)
            
            # Initialize velocities for momentum
            self.v_w.append(np.zeros_like(w))
            self.v_b.append(np.zeros_like(b))
    
    def _relu(self, x):
        # ReLU activation function
        return np.maximum(0, x)
    
    def _relu_derivative(self, x):
        # ReLU derivative
        return (x > 0).astype(float)
    
    def _forward_pass(self, X, training=True):
        # forward propagation
        activations = [X]
        self.dropout_masks = [] # Store masks for backward pass
        
        for i in range(len(self.weights)):
            z = np.dot(activations[-1], self.weights[i]) + self.biases[i]
            if i < len(self.weights) - 1:  # hidden layerlarda Leaky ReLU
                a = self._relu(z)
                
                # Dropout implementation
                if training and self.dropout_rate > 0:
                    mask = (np.random.rand(*a.shape) > self.dropout_rate) / (1.0 - self.dropout_rate)
                    a *= mask
                    self.dropout_masks.append(mask)
                else:
                    self.dropout_masks.append(None)
            else:  # output layerda linear activation (regression icin)
                a = z
            activations.append(a)
        
        return activations
    
    def _backward_pass(self, X, y, activations):
        # backward propagation (gradient descent)
        m = X.shape[0]
        grads_w = []
        grads_b = []
        
        # output layer error
        delta = activations[-1] - y.reshape(-1, 1)
        
        # geriye dogru git
        for i in range(len(self.weights) - 1, -1, -1):
            grad_w = np.dot(activations[i].T, delta) / m
            grad_b = np.sum(delta, axis=0, keepdims=True) / m
            
            grads_w.insert(0, grad_w)
            grads_b.insert(0, grad_b)
            
            if i > 0:  # hidden layerlara devam et
                # Backpropagate through activation derivative
                delta = np.dot(delta, self.weights[i].T) * self._relu_derivative(activations[i])
                
                # Backpropagate through dropout
                if self.dropout_masks[i-1] is not None:
                    delta *= self.dropout_masks[i-1]
        
        return grads_w, grads_b
    
    def fit(self, X, y):
        # modeli train et
        # target'i normalize et (regression icin onemli)
        self.y_mean = np.mean(y)
        self.y_std = np.std(y)
        y_normalized = (y - self.y_mean) / self.y_std
        
        self._initialize_weights(X.shape[1])
        
        n_samples = X.shape[0]
        
        for epoch in range(self.epochs):
            # shuffle data her epoch'da
            indices = np.random.permutation(n_samples)
            X_shuffled = X[indices]
            y_shuffled = y_normalized[indices]
            
            # mini-batch gradient descent
            for i in range(0, n_samples, self.batch_size):
                X_batch = X_shuffled[i:i+self.batch_size]
                y_batch = y_shuffled[i:i+self.batch_size]
                
                # forward pass
                activations = self._forward_pass(X_batch)
                
                # backward pass
                grads_w, grads_b = self._backward_pass(X_batch, y_batch, activations)
                
                # weight update with Momentum and L2
                for j in range(len(self.weights)):
                    # Add L2 regularization gradient: lambda * w
                    grad_w_with_l2 = grads_w[j] + self.l2_lambda * self.weights[j]
                    
                    # Update velocity
                    self.v_w[j] = self.momentum * self.v_w[j] + self.learning_rate * grad_w_with_l2
                    self.v_b[j] = self.momentum * self.v_b[j] + self.learning_rate * grads_b[j]
                    
                    # Update weights
                    self.weights[j] -= self.v_w[j]
                    self.biases[j] -= self.v_b[j]
            
            # Calculate loss for the epoch
            predictions = self._predict_normalized(X)
            mse = np.mean((predictions - y_normalized) ** 2)
            self.loss_history.append(mse)

            # Calculate MAPE on training set for monitoring
            pred_denorm = predictions * self.y_std + self.y_mean
            mask = y != 0
            mape = np.mean(np.abs((y[mask] - pred_denorm[mask]) / y[mask])) * 100

            # her 20 epochta bir loss yazdir
            if (epoch + 1) % 20 == 0:
                print(f"Epoch {epoch+1}/{self.epochs}, Norm. MSE: {mse:.4f}, Train MAPE: {mape:.2f}%")
        
        return self
    
    def _predict_normalized(self, X):
        # normalized prediction (internal use)
        # Disable dropout during prediction
        activations = self._forward_pass(X, training=False)
        return activations[-1].flatten()
    
    def predict(self, X):
        # prediction (denormalized)
        y_normalized = self._predict_normalized(X)
        return y_normalized * self.y_std + self.y_mean

def train_neural_network(X_train, y_train, hidden_layers=[128, 64, 32], learning_rate=0.005, epochs=500, momentum=0.9, l2_lambda=0.0001, batch_size=64, dropout_rate=0.0):
    print(f"Training Custom Neural Network...")
    print(f"Architecture: Input({X_train.shape[1]}) -> Hidden{hidden_layers} -> Output(1)")
    
    model = CustomNeuralNetwork(
        hidden_layers=hidden_layers,
        learning_rate=learning_rate,
        epochs=epochs,
        batch_size=batch_size,
        random_state=42,
        momentum=momentum,
        l2_lambda=l2_lambda,
        dropout_rate=dropout_rate
    )
    
    model.fit(X_train, y_train)
    return model

def evaluate_neural_network(model, X_test, y_test):
    # test seti uzerinde degerlendir
    y_pred = model.predict(X_test)
    
    mse = np.mean((y_test - y_pred) ** 2)
    ss_res = np.sum((y_test - y_pred) ** 2)
    ss_tot = np.sum((y_test - np.mean(y_test)) ** 2)
    r2 = 1 - (ss_res / ss_tot)
    
    return mse, r2, y_pred
