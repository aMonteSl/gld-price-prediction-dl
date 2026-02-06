"""
Evaluation metrics and utilities for model performance assessment.
"""
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


class ModelEvaluator:
    """Evaluate model performance with various metrics."""
    
    @staticmethod
    def evaluate_regression(y_true, y_pred):
        """
        Evaluate regression model performance.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            
        Returns:
            Dictionary of metrics
        """
        # Remove NaN values
        mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
        y_true = y_true[mask]
        y_pred = y_pred[mask]
        
        metrics = {
            'mse': mean_squared_error(y_true, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'mae': mean_absolute_error(y_true, y_pred),
            'r2': r2_score(y_true, y_pred)
        }
        
        return metrics
    
    @staticmethod
    def evaluate_classification(y_true, y_pred, threshold=0.5):
        """
        Evaluate classification model performance.
        
        Args:
            y_true: True labels
            y_pred: Predicted probabilities
            threshold: Classification threshold
            
        Returns:
            Dictionary of metrics
        """
        # Remove NaN values
        mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
        y_true = y_true[mask]
        y_pred = y_pred[mask]
        
        # Convert probabilities to binary predictions
        y_pred_binary = (y_pred > threshold).astype(int)
        
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred_binary),
            'precision': precision_score(y_true, y_pred_binary, zero_division=0),
            'recall': recall_score(y_true, y_pred_binary, zero_division=0),
            'f1': f1_score(y_true, y_pred_binary, zero_division=0)
        }
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred_binary)
        if cm.shape == (2, 2):
            metrics['confusion_matrix'] = cm.tolist()
        
        return metrics
    
    @staticmethod
    def print_metrics(metrics, task='regression'):
        """
        Print metrics in a formatted way.
        
        Args:
            metrics: Dictionary of metrics
            task: 'regression' or 'classification'
        """
        print(f"\n{task.upper()} METRICS:")
        print("=" * 50)
        
        if task == 'regression':
            print(f"MSE:  {metrics['mse']:.6f}")
            print(f"RMSE: {metrics['rmse']:.6f}")
            print(f"MAE:  {metrics['mae']:.6f}")
            print(f"R²:   {metrics['r2']:.6f}")
        else:
            print(f"Accuracy:  {metrics['accuracy']:.4f}")
            print(f"Precision: {metrics['precision']:.4f}")
            print(f"Recall:    {metrics['recall']:.4f}")
            print(f"F1 Score:  {metrics['f1']:.4f}")
            
            if 'confusion_matrix' in metrics:
                print("\nConfusion Matrix:")
                print(f"  {metrics['confusion_matrix']}")
        
        print("=" * 50)
