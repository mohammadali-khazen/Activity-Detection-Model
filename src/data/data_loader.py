from typing import Tuple, List
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import scipy.stats as stats
import pickle

class DataLoader:
    def __init__(self, config: dict):
        self.config = config
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        
    def load_data(self, file_path: str) -> pd.DataFrame:
        """Load and preprocess the raw data."""
        data = pd.read_csv(file_path)
        data = data[['timestamp', 'acc_x_h', 'acc_y_h', 'acc_z_h', 
                     'acc_x_w', 'acc_y_w', 'acc_z_w', 
                     'acc_x_c', 'acc_y_c', 'acc_z_c', 'activity']]
        data = data.dropna()
        
        # Convert acceleration columns to float
        acc_columns = [col for col in data.columns if col.startswith('acc_')]
        for col in acc_columns:
            data[col] = data[col].astype('float')
            
        return data
    
    def balance_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Balance the dataset by sampling equal number of samples from each class."""
        # Sample sizes for different activities
        sample_sizes = {
            'idling': 1342,
            'walking': 1342,
            'value_add_work': 2255,
            'non_value_add_work': 4510
        }
        
        balanced_data = pd.DataFrame()
        
        # Sample each activity
        for activity, size in sample_sizes.items():
            if activity == 'value_add_work':
                activity_data = data[data['activity'] == 'painting'].tail(size)
            elif activity == 'non_value_add_work':
                # Combine multiple activities for non-value-add work
                activities = ['shoveling', 'mortar', 'brick_laying', 'plastering']
                activity_data = pd.concat([
                    data[data['activity'] == act].tail(size // len(activities))
                    for act in activities
                ])
            else:
                activity_data = data[data['activity'] == activity].tail(size)
            
            activity_data['activity'] = activity
            balanced_data = pd.concat([balanced_data, activity_data])
            
        return balanced_data
    
    def prepare_frames(self, data: pd.DataFrame, frame_size: int, hop_size: int) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare frames from the time series data."""
        frames = []
        labels = []
        
        for i in range(0, len(data) - frame_size, hop_size):
            # Extract features for the current frame
            frame_features = []
            for col in ['acc_x_h', 'acc_y_h', 'acc_z_h', 
                       'acc_x_w', 'acc_y_w', 'acc_z_w',
                       'acc_x_c', 'acc_y_c', 'acc_z_c']:
                frame_features.append(data[col].values[i:i + frame_size])
            
            # Get the most common label in this segment
            label = stats.mode(data['label'][i:i + frame_size])[0][0]
            
            frames.append(frame_features)
            labels.append(label)
            
        return np.asarray(frames), np.asarray(labels)
    
    def preprocess_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Preprocess the data by calculating differences and handling missing values."""
        acc_columns = [col for col in data.columns if col.startswith('acc_')]
        for col in acc_columns:
            data[col] = data[col].diff(1)
        return data.fillna(0)
    
    def prepare_training_data(self, file_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Prepare the complete training pipeline."""
        # Load and preprocess data
        data = self.load_data(file_path)
        data = self.balance_data(data)
        
        # Encode labels
        data['label'] = self.label_encoder.fit_transform(data['activity'])
        
        # Preprocess features
        data = self.preprocess_data(data)
        
        # Prepare frames
        X, y = self.prepare_frames(data, 
                                 self.config['frame_size'], 
                                 self.config['hop_size'])
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=self.config['test_size'],
            random_state=self.config['random_state'],
            stratify=y
        )
        
        # Scale features
        X_train = self.scaler.fit_transform(
            X_train.reshape(-1, X_train.shape[-1])
        ).reshape(X_train.shape)
        
        X_test = self.scaler.transform(
            X_test.reshape(-1, X_test.shape[-1])
        ).reshape(X_test.shape)
        
        # Reshape for CNN
        X_train = X_train.reshape(X_train.shape[0], 9, 4, 1)
        X_test = X_test.reshape(X_test.shape[0], 9, 4, 1)
        
        return X_train, X_test, y_train, y_test
    
    def save_scaler(self, file_path: str):
        """Save the fitted scaler."""
        pickle.dump(self.scaler, open(file_path, 'wb')) 