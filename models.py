"""
PyTorch model architectures for GLD price prediction.
"""
import torch
import torch.nn as nn


class GRURegressor(nn.Module):
    """GRU-based model for regression (returns prediction)."""
    
    def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.2):
        """
        Initialize GRU regressor.
        
        Args:
            input_size: Number of input features
            hidden_size: Size of hidden state
            num_layers: Number of GRU layers
            dropout: Dropout probability
        """
        super(GRURegressor, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1)
        )
        
    def forward(self, x):
        """Forward pass."""
        # x shape: (batch, seq_len, features)
        gru_out, _ = self.gru(x)
        # Take output from last time step
        last_output = gru_out[:, -1, :]
        output = self.fc(last_output)
        return output.squeeze()


class LSTMRegressor(nn.Module):
    """LSTM-based model for regression (returns prediction)."""
    
    def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.2):
        """
        Initialize LSTM regressor.
        
        Args:
            input_size: Number of input features
            hidden_size: Size of hidden state
            num_layers: Number of LSTM layers
            dropout: Dropout probability
        """
        super(LSTMRegressor, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1)
        )
        
    def forward(self, x):
        """Forward pass."""
        # x shape: (batch, seq_len, features)
        lstm_out, _ = self.lstm(x)
        # Take output from last time step
        last_output = lstm_out[:, -1, :]
        output = self.fc(last_output)
        return output.squeeze()


class GRUClassifier(nn.Module):
    """GRU-based model for classification (buy/no-buy signals)."""
    
    def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.2):
        """
        Initialize GRU classifier.
        
        Args:
            input_size: Number of input features
            hidden_size: Size of hidden state
            num_layers: Number of GRU layers
            dropout: Dropout probability
        """
        super(GRUClassifier, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        """Forward pass."""
        # x shape: (batch, seq_len, features)
        gru_out, _ = self.gru(x)
        # Take output from last time step
        last_output = gru_out[:, -1, :]
        output = self.fc(last_output)
        return output.squeeze()


class LSTMClassifier(nn.Module):
    """LSTM-based model for classification (buy/no-buy signals)."""
    
    def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.2):
        """
        Initialize LSTM classifier.
        
        Args:
            input_size: Number of input features
            hidden_size: Size of hidden state
            num_layers: Number of LSTM layers
            dropout: Dropout probability
        """
        super(LSTMClassifier, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        """Forward pass."""
        # x shape: (batch, seq_len, features)
        lstm_out, _ = self.lstm(x)
        # Take output from last time step
        last_output = lstm_out[:, -1, :]
        output = self.fc(last_output)
        return output.squeeze()
