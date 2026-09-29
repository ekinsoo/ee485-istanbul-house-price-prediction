import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from data_preparation import load_and_clean_data, prepare_data, load_and_parse_data
from ridge_regression import train_ridge, evaluate_ridge
from neural_network import train_neural_network, evaluate_neural_network
from decision_tree import train_decision_tree, evaluate_decision_tree
def calculate_mape(y_true, y_pred):
    # Mean Absolute Percentage Error
    # Avoid division by zero
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

def calculate_mpe(y_true, y_pred):
    # Mean Percentage Error
    mask = y_true != 0
    return np.mean((y_true[mask] - y_pred[mask]) / y_true[mask]) * 100

def calculate_nmse(y_true, y_pred):
    # Normalized Mean Squared Error
    mse = np.mean((y_true - y_pred) ** 2)
    var = np.var(y_true)
    return mse / var

def optimize_neural_network(X_train, y_train, X_val, y_val):
    print("Optimizing Neural Network Hyperparameters...")
    learning_rates = [0.001, 0.005, 0.01]
    architectures = [[64, 32], [128, 64, 32]]
    best_r2 = -np.inf
    best_params = {}
    best_model = None

    for lr in learning_rates:
        for arch in architectures:
            print(f"Testing LR={lr}, Arch={arch}")
            # Reduced epochs for search
            model = train_neural_network(X_train, y_train, hidden_layers=arch, learning_rate=lr, epochs=100)
            _, r2, _ = evaluate_neural_network(model, X_val, y_val)
            print(f"Result: R2={r2:.4f}")
            
            if r2 > best_r2:
                best_r2 = r2
                best_params = {'learning_rate': lr, 'hidden_layers': arch}
                best_model = model
    
    print(f"Best NN Params: {best_params}")
    return best_params

def optimize_decision_tree(X_train, y_train, X_val, y_val):
    print("Optimizing Decision Tree Hyperparameters...")
    depths = [10, 15, 20]
    best_r2 = -np.inf
    best_depth = 10
    
    for depth in depths:
        print(f"Testing Max Depth={depth}")
        model = train_decision_tree(X_train, y_train, max_depth=depth)
        _, r2, _ = evaluate_decision_tree(model, X_val, y_val)
        print(f"Result: R2={r2:.4f}")
        
        if r2 > best_r2:
            best_r2 = r2
            best_depth = depth
            
    print(f"Best DT Depth: {best_depth}")
    return best_depth

def main():
    project_dir = Path(__file__).resolve().parent
    data_path = project_dir / 'HouseData.csv'
    processed_data_path = project_dir / 'processed_data.npz'
    report_dir = project_dir / 'report'
    report_dir.mkdir(exist_ok=True)
    
    #data preparation part
    if processed_data_path.exists():
        print("Loading processed data")
        data = np.load(processed_data_path, allow_pickle=True)
        X_train = data['X_train']
        X_test = data['X_test']
        y_train = data['y_train']
        y_test = data['y_test']
        feature_names = data['feature_names']
    else:
        print("Processing data from scratch...")
        df = load_and_clean_data(data_path)
        X_train, X_test, y_train, y_test, feature_names = prepare_data(df)
        np.savez(processed_data_path, X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test, feature_names=feature_names)
# print dataset information
    print(f"Training samples: {X_train.shape[0]}, Test samples: {X_test.shape[0]}")
    print(f"Number of features: {X_train.shape[1]}")

    results = {}

    #ridge regression with hyperparameter tuning (degisik alpha degerleri)
    #note: We tried log-transformation but it reduced performance (R2 ~0.46), so we stick to raw target with optimized alpha.
    print("\n=== Ridge Regression ===")
    ridge_model = train_ridge(X_train, y_train)
    #ridge function
    mse_ridge, r2_ridge, pred_ridge = evaluate_ridge(ridge_model, X_test, y_test)
    mape_ridge = calculate_mape(y_test, pred_ridge)
    mpe_ridge = calculate_mpe(y_test, pred_ridge)
    nmse_ridge = calculate_nmse(y_test, pred_ridge)
    
    results['Ridge'] = {'MSE': mse_ridge, 'R2': r2_ridge, 'MAPE': mape_ridge, 'MPE': mpe_ridge, 'NMSE': nmse_ridge}
    print(f"Ridge Regression - MSE: {mse_ridge:.2f}, R2: {r2_ridge:.4f}, MAPE: {mape_ridge:.2f}%, MPE: {mpe_ridge:.2f}%, NMSE: {nmse_ridge:.4f}")

    # Neural Network
    print("\n=== Neural Network ===")
    # Reverting to the configuration that yielded ~0.72 R2
    nn_model = train_neural_network(
        X_train, y_train, 
        hidden_layers=[128, 64, 32], 
        learning_rate=0.005, 
        epochs=300,
        momentum=0.0,
        l2_lambda=0.0,
        batch_size=85,
        dropout_rate=0.0
    )
    mse_nn, r2_nn, pred_nn = evaluate_neural_network(nn_model, X_test, y_test)
    mape_nn = calculate_mape(y_test, pred_nn)
    mpe_nn = calculate_mpe(y_test, pred_nn)
    nmse_nn = calculate_nmse(y_test, pred_nn)
    
    results['Neural Network'] = {'MSE': mse_nn, 'R2': r2_nn, 'MAPE': mape_nn, 'MPE': mpe_nn, 'NMSE': nmse_nn}
    print(f"Neural Network - MSE: {mse_nn:.2f}, R2: {r2_nn:.4f}, MAPE: {mape_nn:.2f}%, MPE: {mpe_nn:.2f}%, NMSE: {nmse_nn:.4f}")

    # Plot Learning Curve
    if hasattr(nn_model, 'loss_history') and nn_model.loss_history:
        plt.figure(figsize=(10, 6))
        plt.plot(nn_model.loss_history)
        plt.title('Neural Network Training Loss (Normalized MSE)')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True, alpha=0.3)
        plt.savefig(report_dir / 'loss_curve.png')
        print("Loss curve saved to report/loss_curve.png")

    # Decision Tree
    print("\n=== Decision Tree ===")
    # Using best depth found: 10
    dt_model = train_decision_tree(X_train, y_train, max_depth=10)
    mse_dt, r2_dt, pred_dt = evaluate_decision_tree(dt_model, X_test, y_test)
    mape_dt = calculate_mape(y_test, pred_dt)
    mpe_dt = calculate_mpe(y_test, pred_dt)
    nmse_dt = calculate_nmse(y_test, pred_dt)
    
    results['Decision Tree'] = {'MSE': mse_dt, 'R2': r2_dt, 'MAPE': mape_dt, 'MPE': mpe_dt, 'NMSE': nmse_dt}
    print(f"Decision Tree - MSE: {mse_dt:.2f}, R2: {r2_dt:.4f}, MAPE: {mape_dt:.2f}%, MPE: {mpe_dt:.2f}%, NMSE: {nmse_dt:.4f}")

    # coefficients for feature importance
    if hasattr(ridge_model, 'coef_'):
        coefs = ridge_model.coef_
        #sorting and plotting part 
        indices = np.argsort(np.abs(coefs))[::-1]
        top_n = 20
        top_indices = indices[:top_n]
        
        plt.figure(figsize=(12, 6))
        plt.barh(range(top_n), coefs[top_indices], align='center')
        plt.yticks(range(top_n), [feature_names[i] for i in top_indices])
        plt.xlabel('Coefficient Value')
        plt.title('Top 20 Feature Importances (Ridge Regression)')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.savefig(report_dir / 'feature_importance.png')
        print("Feature importance plot saved to report/feature_importance.png")

    #price distribution
    plt.figure(figsize=(10, 6))
    plt.hist(y_train, bins=50, color='blue', alpha=0.7, label='Train')
    plt.hist(y_test, bins=50, color='red', alpha=0.7, label='Test')
    plt.title('Distribution of House Prices')
    plt.xlabel('Price (TL)')
    plt.ylabel('Frequency')
    plt.legend()
    plt.savefig(report_dir / 'price_distribution.png')
    print("Price distribution plot saved to report/price_distribution.png")
    
    # Outlier Analysis Plot (Before vs After)
    print("Generating outlier analysis plot...")
    df_raw = load_and_parse_data(data_path)
    prices_raw = df_raw['price'].values
    prices_clean = np.concatenate([y_train, y_test]) # Reconstruct full cleaned dataset
    
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    # Use a wider range for raw data to show the presence of outliers
    # We limit it to 2x the max of cleaned data to keep the main distribution visible
    limit_raw = np.max(prices_clean) * 2.0
    plt.hist(prices_raw, bins=50, range=(0, limit_raw), color='gray', alpha=0.7)
    plt.title('Before Outlier Removal (Clipped at 2x Clean Max)')
    plt.xlabel('Price (TL)')
    plt.ylabel('Frequency')
    
    plt.subplot(1, 2, 2)
    plt.hist(prices_clean, bins=50, color='blue', alpha=0.7)
    plt.title('After Outlier Removal (IQR)')
    plt.xlabel('Price (TL)')
    plt.ylabel('Frequency')
    
    plt.tight_layout()
    plt.savefig(report_dir / 'outlier_analysis.png')
    print("Outlier analysis plot saved to report/outlier_analysis.png")

    # Model comparison bar chart
    print("\n--- Final Results ---")
    models = list(results.keys())
    r2_scores = [results[m]['R2'] for m in models]
    mse_scores = [results[m]['MSE'] for m in models]
    mape_scores = [results[m]['MAPE'] for m in models]
    
    for model_name in models:
        print(f"{model_name}: MSE={results[model_name]['MSE']:.2f}, R2={results[model_name]['R2']:.4f}, MAPE={results[model_name]['MAPE']:.2f}%")
    
    # R2 comparison plot
    plt.figure(figsize=(10, 6))
    bars = plt.bar(models, r2_scores, color=['blue', 'green', 'orange'])
    plt.ylabel('R² Score')
    plt.title('Model Performance Comparison (R² Score)')
    plt.ylim([0, 1])
    for i, bar in enumerate(bars):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{r2_scores[i]:.4f}',
                ha='center', va='bottom')
    plt.tight_layout()
    plt.savefig(report_dir / 'model_comparison.png')
    print("Model comparison plot saved to report/model_comparison.png")
    
    print("\n--- Final Results ---")
    print(f"Ridge Regression: MSE={mse_ridge:.2f}, R2={r2_ridge:.4f}, MAPE={mape_ridge:.2f}%")

    # Scatter Plot (Actual vs Predicted)
    plt.figure(figsize=(10, 10))
    plt.scatter(y_test, pred_nn, alpha=0.5, color='green', label='Neural Network')
    plt.scatter(y_test, pred_ridge, alpha=0.3, color='blue', label='Ridge', marker='x')
    plt.scatter(y_test, pred_dt, alpha=0.3, color='orange', label='Decision Tree', marker='^')
    
    # Perfect prediction line
    min_val = min(np.min(y_test), np.min(pred_nn))
    max_val = max(np.max(y_test), np.max(pred_nn))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
    
    plt.title('Actual vs Predicted Prices')
    plt.xlabel('Actual Price (TL)')
    plt.ylabel('Predicted Price (TL)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(report_dir / 'scatter_plot.png')
    print("Scatter plot saved to report/scatter_plot.png")

    #predictions vs actial results just for 50 samples
    plt.figure(figsize=(15, 8))
    subset = 50
    x_ax = range(subset)
    
    plt.plot(x_ax, y_test[:subset], label="Actual", color='black', linestyle='-', linewidth=2, marker='o')
    plt.plot(x_ax, pred_ridge[:subset], label="Ridge Prediction", linestyle='--', color='blue', marker='s')
    plt.plot(x_ax, pred_nn[:subset], label="Neural Network Prediction", linestyle='--', color='green', marker='^')
    plt.plot(x_ax, pred_dt[:subset], label="Decision Tree Prediction", linestyle='--', color='orange', marker='D')
    
    plt.title("Model Predictions vs Actual Prices (First 50 samples)")
    plt.xlabel("Sample Index")
    plt.ylabel("Price (TL)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(report_dir / 'comparison_plot.png')
    print("Comparison plot saved to report/comparison_plot.png")

if __name__ == "__main__":
    main()
