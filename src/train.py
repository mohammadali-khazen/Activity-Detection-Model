import os
from typing import Dict, Any
import numpy as np

from config.config import get_config
from data.data_loader import DataLoader
from models.cnn_model import ActivityDetectionModel

def create_directories(config: Dict[str, Any]):
    """Create necessary directories for saving models and plots."""
    os.makedirs(os.path.dirname(config['model_save_path']), exist_ok=True)
    os.makedirs(os.path.dirname(config['scaler_save_path']), exist_ok=True)
    os.makedirs(config['plots_save_dir'], exist_ok=True)

def main():
    # Load configuration
    config = get_config()
    
    # Create necessary directories
    create_directories(config)
    
    # Initialize data loader
    data_loader = DataLoader(config)
    
    # Load and preprocess data
    print("Loading and preprocessing data...")
    X_train, X_test, y_train, y_test = data_loader.prepare_training_data(config['data_path'])
    
    # Save the fitted scaler
    data_loader.save_scaler(config['scaler_save_path'])
    
    # Initialize and train model
    print("Initializing model...")
    model = ActivityDetectionModel(config)
    
    print("Training model...")
    history = model.train(X_train, y_train, X_test, y_test)
    
    # Evaluate model
    print("\nEvaluating model...")
    metrics = model.evaluate(X_test, y_test)
    print(f"Test accuracy: {metrics['accuracy']:.4f}")
    print(f"Test loss: {metrics['loss']:.4f}")
    
    # Plot training history
    print("\nPlotting training history...")
    model.plot_training_history(
        history,
        save_path=os.path.join(config['plots_save_dir'], 'training_history.png')
    )
    
    # Plot confusion matrix
    print("\nPlotting confusion matrix...")
    model.plot_confusion_matrix(
        X_test, y_test,
        class_names=['idling', 'walking', 'value_add_work', 'non_value_add_work'],
        save_path=os.path.join(config['plots_save_dir'], 'confusion_matrix.png')
    )

if __name__ == "__main__":
    main() 