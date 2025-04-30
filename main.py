
from src.preprocess import load_data, preprocess_data
from src.models import initialize_models, build_dnn
from src.train import train_model, train_dnn
from src.evaluate import evaluate_model, plot_comparison_bars
from src.config import RESULTS_DIR
import pandas as pd

def main():
    # Load and preprocess data
    X, y = load_data()
    X_train, X_test, y_train, y_test, scaler = preprocess_data(X, y)
    
    # Initialize models
    models = initialize_models()
    
    # Train and evaluate
    results = []
    for name, model in models.items():
        print(f'Training {name}...')
        if name == 'DNN':
            history = train_dnn(model, X_train, y_train)
        else:
            trained_model = train_model(model, X_train, y_train)
        
        metrics, _ = evaluate_model(trained_model, X_test, y_test, name)
        results.append({'Model': name, **metrics})
    
    # Save final metrics
    metrics_path = f'{RESULTS_DIR}/metrics_summary.csv'
    pd.DataFrame(results).to_csv(metrics_path, index=False)

    # Plot comparison bar graphs for key metrics
    plot_comparison_bars(metrics_path)
if __name__ == '__main__':
    main()

