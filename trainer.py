"""
Training pipeline for GLD price prediction models.
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import os


class ModelTrainer:
    """Train and evaluate PyTorch models."""
    
    def __init__(self, model, task='regression', device=None):
        """
        Initialize trainer.
        
        Args:
            model: PyTorch model
            task: 'regression' or 'classification'
            device: torch device (auto-detected if None)
        """
        self.model = model
        self.task = task
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        # Initialize loss function
        if task == 'regression':
            self.criterion = nn.MSELoss()
        else:
            self.criterion = nn.BCELoss()
        
        self.scaler = StandardScaler()
        self.history = {'train_loss': [], 'val_loss': []}
        
    def prepare_data(self, X, y, test_size=0.2, batch_size=32):
        """
        Prepare data for training.
        
        Args:
            X: Feature sequences (numpy array)
            y: Target values (numpy array)
            test_size: Fraction of data for validation
            batch_size: Batch size for training
            
        Returns:
            Tuple of (train_loader, val_loader)
        """
        # Remove NaN values
        valid_indices = ~np.isnan(y)
        X = X[valid_indices]
        y = y[valid_indices]
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=test_size, shuffle=False
        )
        
        # Normalize features
        n_samples, n_timesteps, n_features = X_train.shape
        X_train_reshaped = X_train.reshape(-1, n_features)
        X_val_reshaped = X_val.reshape(-1, n_features)
        
        X_train_scaled = self.scaler.fit_transform(X_train_reshaped)
        X_val_scaled = self.scaler.transform(X_val_reshaped)
        
        X_train = X_train_scaled.reshape(n_samples, n_timesteps, n_features)
        X_val = X_val_scaled.reshape(-1, n_timesteps, n_features)
        
        # Convert to tensors
        X_train_tensor = torch.FloatTensor(X_train)
        y_train_tensor = torch.FloatTensor(y_train)
        X_val_tensor = torch.FloatTensor(X_val)
        y_val_tensor = torch.FloatTensor(y_val)
        
        # Create data loaders
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        return train_loader, val_loader
    
    def train(self, train_loader, val_loader, epochs=50, learning_rate=0.001):
        """
        Train the model.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            epochs: Number of training epochs
            learning_rate: Learning rate for optimizer
            
        Returns:
            Training history dictionary
        """
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        
        for epoch in range(epochs):
            # Training phase
            self.model.train()
            train_loss = 0.0
            
            for batch_X, batch_y in train_loader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)
                
                # Forward pass
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = self.criterion(outputs, batch_y)
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            train_loss /= len(train_loader)
            
            # Validation phase
            self.model.eval()
            val_loss = 0.0
            
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    batch_X = batch_X.to(self.device)
                    batch_y = batch_y.to(self.device)
                    
                    outputs = self.model(batch_X)
                    loss = self.criterion(outputs, batch_y)
                    val_loss += loss.item()
            
            val_loss /= len(val_loader)
            
            # Record history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            
            if (epoch + 1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{epochs}], Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}')
        
        return self.history
    
    def predict(self, X):
        """
        Make predictions on new data.
        
        Args:
            X: Input sequences (numpy array)
            
        Returns:
            Predictions as numpy array
        """
        self.model.eval()
        
        # Normalize features
        n_samples, n_timesteps, n_features = X.shape
        X_reshaped = X.reshape(-1, n_features)
        X_scaled = self.scaler.transform(X_reshaped)
        X = X_scaled.reshape(n_samples, n_timesteps, n_features)
        
        X_tensor = torch.FloatTensor(X).to(self.device)
        
        with torch.no_grad():
            predictions = self.model(X_tensor)
        
        return predictions.cpu().numpy()
    
    def save_model(self, filepath):
        """Save model and scaler."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'scaler': self.scaler,
            'task': self.task
        }, filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        """Load model and scaler."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.scaler = checkpoint['scaler']
        self.task = checkpoint['task']
        print(f"Model loaded from {filepath}")
