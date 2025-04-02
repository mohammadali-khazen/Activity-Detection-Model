from typing import Dict, Any

def get_config() -> Dict[str, Any]:
    """Get the configuration dictionary with all hyperparameters and settings."""
    return {
        # Data parameters
        'frame_size': 4,
        'hop_size': 1,
        'test_size': 0.1,
        'random_state': 42,
        
        # Model parameters
        'learning_rate': 0.001,
        'epochs': 50,
        'early_stopping_patience': 10,
        
        # File paths
        'data_path': 'data/labelled_dataset.csv',
        'model_save_path': 'models/activity_detection_model.h5',
        'scaler_save_path': 'models/scaler.pkl',
        'plots_save_dir': 'plots/',
        
        # Model architecture
        'conv_layers': [
            {'filters': 32, 'kernel_size': (2, 2), 'dropout': 0.2},
            {'filters': 32, 'kernel_size': (2, 2), 'dropout': 0.2},
            {'filters': 32, 'kernel_size': (2, 2), 'dropout': 0.2}
        ],
        'dense_layers': [
            {'units': 64, 'dropout': 0.2},
            {'units': 4, 'activation': 'softmax'}
        ]
    } 