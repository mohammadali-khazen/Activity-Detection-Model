from typing import Dict, Any
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Flatten, Dense, Dropout, Conv2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
from mlxtend.plotting import plot_confusion_matrix

class ActivityDetectionModel:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model = self._build_model()
        
    def _build_model(self) -> Sequential:
        """Build the CNN model architecture."""
        model = Sequential([
            Conv2D(32, (2, 2), activation='relu', input_shape=(9, 4, 1)),
            Dropout(0.2),
            
            Conv2D(32, (2, 2), activation='relu'),
            Dropout(0.2),
            
            Conv2D(32, (2, 2), activation='relu'),
            Dropout(0.2),
            
            Flatten(),
            
            Dense(64, activation='relu'),
            Dropout(0.2),
            
            Dense(4, activation='softmax')
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=self.config['learning_rate']),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: np.ndarray, y_val: np.ndarray) -> Dict[str, Any]:
        """Train the model with early stopping and model checkpointing."""
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=self.config['early_stopping_patience'],
                restore_best_weights=True
            ),
            ModelCheckpoint(
                filepath=self.config['model_save_path'],
                monitor='val_accuracy',
                save_best_only=True,
                mode='max'
            )
        ]
        
        history = self.model.fit(
            X_train, y_train,
            epochs=self.config['epochs'],
            validation_data=(X_val, y_val),
            callbacks=callbacks,
            verbose=1
        )
        
        return history.history
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """Evaluate the model on test data."""
        return dict(zip(self.model.metrics_names, self.model.evaluate(X_test, y_test)))
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions on input data."""
        return self.model.predict(X)
    
    def plot_training_history(self, history: Dict[str, Any], save_path: str = None):
        """Plot training and validation metrics."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot accuracy
        ax1.plot(history['accuracy'], label='Training')
        ax1.plot(history['val_accuracy'], label='Validation')
        ax1.set_title('Model Accuracy')
        ax1.set_ylabel('Accuracy')
        ax1.set_xlabel('Epoch')
        ax1.legend()
        
        # Plot loss
        ax2.plot(history['loss'], label='Training')
        ax2.plot(history['val_loss'], label='Validation')
        ax2.set_title('Model Loss')
        ax2.set_ylabel('Loss')
        ax2.set_xlabel('Epoch')
        ax2.legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
        plt.show()
    
    def plot_confusion_matrix(self, X_test: np.ndarray, y_test: np.ndarray,
                            class_names: list, save_path: str = None):
        """Plot confusion matrix for model predictions."""
        y_pred = self.predict(X_test)
        y_pred = np.argmax(y_pred, axis=1)
        
        mat = confusion_matrix(y_test, y_pred)
        plot_confusion_matrix(
            conf_mat=mat,
            class_names=class_names,
            show_normed=True,
            figsize=(8, 8)
        )
        
        if save_path:
            plt.savefig(save_path)
        plt.show()
        
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=class_names)) 