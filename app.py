"""
Streamlit GUI for GLD price prediction application.

Run with: streamlit run app.py
"""
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from datetime import datetime, timedelta
import torch
import os

from data_loader import GLDDataLoader
from feature_engineering import FeatureEngineering
from models import GRURegressor, LSTMRegressor, GRUClassifier, LSTMClassifier
from trainer import ModelTrainer
from evaluator import ModelEvaluator


# Page configuration
st.set_page_config(
    page_title="GLD Price Prediction",
    page_icon="📈",
    layout="wide"
)

st.title("🏅 GLD Price Prediction with Deep Learning")
st.markdown("Forecast Gold ETF price movements using PyTorch GRU/LSTM models")

# Sidebar configuration
st.sidebar.header("Configuration")

# Data settings
st.sidebar.subheader("Data Settings")
start_date = st.sidebar.date_input(
    "Start Date",
    value=datetime.now() - timedelta(days=365*5)
)
end_date = st.sidebar.date_input(
    "End Date",
    value=datetime.now()
)

# Model settings
st.sidebar.subheader("Model Settings")
model_type = st.sidebar.selectbox(
    "Model Architecture",
    ["GRU", "LSTM"]
)

task_type = st.sidebar.selectbox(
    "Task Type",
    ["Regression (Returns)", "Classification (Buy/No-Buy)"]
)

horizon = st.sidebar.selectbox(
    "Prediction Horizon (days)",
    [1, 5, 20]
)

# Training settings
st.sidebar.subheader("Training Settings")
seq_length = st.sidebar.slider("Sequence Length", 10, 60, 20)
hidden_size = st.sidebar.slider("Hidden Size", 32, 128, 64)
num_layers = st.sidebar.slider("Number of Layers", 1, 4, 2)
epochs = st.sidebar.slider("Epochs", 10, 200, 50)
batch_size = st.sidebar.slider("Batch Size", 16, 128, 32)
learning_rate = st.sidebar.select_slider(
    "Learning Rate",
    options=[0.0001, 0.0005, 0.001, 0.005, 0.01],
    value=0.001
)

# Initialize session state
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'model_trained' not in st.session_state:
    st.session_state.model_trained = False

# Main content tabs
tab1, tab2, tab3, tab4 = st.tabs(["📊 Data", "🔧 Train Model", "📈 Predictions", "📉 Evaluation"])

# Tab 1: Data Loading and Exploration
with tab1:
    st.header("Data Loading and Exploration")
    
    if st.button("Load Data"):
        with st.spinner("Loading GLD data..."):
            try:
                loader = GLDDataLoader(start_date=start_date, end_date=end_date)
                data = loader.load_data()
                
                # Feature engineering
                fe = FeatureEngineering()
                data_with_features = fe.add_technical_indicators(data)
                
                st.session_state.loader = loader
                st.session_state.data = data
                st.session_state.data_with_features = data_with_features
                st.session_state.data_loaded = True
                
                st.success(f"✅ Loaded {len(data)} records from {start_date} to {end_date}")
            except Exception as e:
                st.error(f"❌ Error loading data: {str(e)}")
    
    if st.session_state.data_loaded:
        data = st.session_state.data
        data_with_features = st.session_state.data_with_features
        
        # Display basic statistics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Records", len(data))
        with col2:
            st.metric("Latest Price", f"${data['Close'].iloc[-1]:.2f}")
        with col3:
            st.metric("Price Change", f"{((data['Close'].iloc[-1] / data['Close'].iloc[0] - 1) * 100):.2f}%")
        with col4:
            st.metric("Features", len(data_with_features.columns))
        
        # Price chart
        st.subheader("Price History")
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=data.index,
            y=data['Close'],
            mode='lines',
            name='Close Price',
            line=dict(color='gold', width=2)
        ))
        fig.update_layout(
            xaxis_title="Date",
            yaxis_title="Price (USD)",
            hovermode='x unified',
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Data preview
        st.subheader("Data Preview")
        st.dataframe(data_with_features.tail(10), use_container_width=True)

# Tab 2: Model Training
with tab2:
    st.header("Model Training")
    
    if not st.session_state.data_loaded:
        st.warning("⚠️ Please load data first in the Data tab")
    else:
        if st.button("Train Model"):
            with st.spinner("Training model..."):
                try:
                    data_with_features = st.session_state.data_with_features
                    loader = st.session_state.loader
                    
                    # Prepare features and targets
                    fe = FeatureEngineering()
                    features = fe.select_features(data_with_features)
                    features = features.ffill().bfill()
                    
                    task = 'regression' if 'Regression' in task_type else 'classification'
                    
                    if task == 'regression':
                        targets = loader.compute_returns(horizon=horizon)
                    else:
                        targets = loader.compute_signals(horizon=horizon)
                    
                    # Create sequences
                    X, y = fe.create_sequences(features, targets, seq_length=seq_length)
                    
                    # Initialize model
                    input_size = X.shape[2]
                    
                    if task == 'regression':
                        if model_type == 'GRU':
                            model = GRURegressor(input_size, hidden_size, num_layers)
                        else:
                            model = LSTMRegressor(input_size, hidden_size, num_layers)
                    else:
                        if model_type == 'GRU':
                            model = GRUClassifier(input_size, hidden_size, num_layers)
                        else:
                            model = LSTMClassifier(input_size, hidden_size, num_layers)
                    
                    # Train model
                    trainer = ModelTrainer(model, task=task)
                    train_loader, val_loader = trainer.prepare_data(X, y, batch_size=batch_size)
                    
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    history = trainer.train(
                        train_loader, val_loader,
                        epochs=epochs,
                        learning_rate=learning_rate
                    )
                    
                    progress_bar.progress(100)
                    status_text.text("Training complete!")
                    
                    # Save to session state
                    st.session_state.model = model
                    st.session_state.trainer = trainer
                    st.session_state.history = history
                    st.session_state.X = X
                    st.session_state.y = y
                    st.session_state.task = task
                    st.session_state.model_trained = True
                    
                    # Save model
                    os.makedirs('models', exist_ok=True)
                    model_path = f'models/{model_type}_{task}_h{horizon}.pth'
                    trainer.save_model(model_path)
                    
                    st.success(f"✅ Model trained successfully! Saved to {model_path}")
                    
                    # Plot training history
                    st.subheader("Training History")
                    fig, ax = plt.subplots(figsize=(10, 4))
                    ax.plot(history['train_loss'], label='Train Loss')
                    ax.plot(history['val_loss'], label='Validation Loss')
                    ax.set_xlabel('Epoch')
                    ax.set_ylabel('Loss')
                    ax.set_title('Training and Validation Loss')
                    ax.legend()
                    ax.grid(True, alpha=0.3)
                    st.pyplot(fig)
                    
                except Exception as e:
                    st.error(f"❌ Error training model: {str(e)}")
                    import traceback
                    st.code(traceback.format_exc())

# Tab 3: Predictions
with tab3:
    st.header("Model Predictions")
    
    if not st.session_state.model_trained:
        st.warning("⚠️ Please train a model first in the Train Model tab")
    else:
        try:
            trainer = st.session_state.trainer
            X = st.session_state.X
            y = st.session_state.y
            task = st.session_state.task
            data = st.session_state.data
            
            # Make predictions
            predictions = trainer.predict(X)
            
            # Align with dates
            num_predictions = len(predictions)
            dates = data.index[-num_predictions:]
            actual_prices = data['Close'].iloc[-num_predictions:].values
            
            # Display predictions
            st.subheader("Predictions vs Actual")
            
            if task == 'regression':
                # For regression, show predicted returns and implied prices
                st.write("**Predicted Returns:**")
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=dates,
                    y=y[-num_predictions:],
                    mode='lines',
                    name='Actual Returns',
                    line=dict(color='blue')
                ))
                fig.add_trace(go.Scatter(
                    x=dates,
                    y=predictions,
                    mode='lines',
                    name='Predicted Returns',
                    line=dict(color='red', dash='dash')
                ))
                fig.update_layout(
                    xaxis_title="Date",
                    yaxis_title="Returns",
                    hovermode='x unified',
                    height=400
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Compute implied prices
                st.write("**Implied Price Movements:**")
                implied_prices = actual_prices * (1 + predictions)
                
                fig2 = go.Figure()
                fig2.add_trace(go.Scatter(
                    x=dates,
                    y=actual_prices,
                    mode='lines',
                    name='Actual Price',
                    line=dict(color='gold', width=2)
                ))
                fig2.add_trace(go.Scatter(
                    x=dates,
                    y=implied_prices,
                    mode='lines',
                    name='Implied Price (from prediction)',
                    line=dict(color='red', dash='dash')
                ))
                fig2.update_layout(
                    xaxis_title="Date",
                    yaxis_title="Price (USD)",
                    hovermode='x unified',
                    height=400
                )
                st.plotly_chart(fig2, use_container_width=True)
                
            else:
                # For classification, show signals
                st.write("**Buy/No-Buy Signals:**")
                
                signals_actual = y[-num_predictions:].astype(int)
                signals_pred = (predictions > 0.5).astype(int)
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=dates,
                    y=signals_actual,
                    mode='markers',
                    name='Actual Signal',
                    marker=dict(size=8, color='blue')
                ))
                fig.add_trace(go.Scatter(
                    x=dates,
                    y=signals_pred,
                    mode='markers',
                    name='Predicted Signal',
                    marker=dict(size=8, color='red', symbol='x')
                ))
                fig.update_layout(
                    xaxis_title="Date",
                    yaxis_title="Signal (1=Buy, 0=No-Buy)",
                    hovermode='x unified',
                    height=400
                )
                st.plotly_chart(fig, use_container_width=True)
                
            # Recent predictions table
            st.subheader("Recent Predictions")
            recent_df = pd.DataFrame({
                'Date': dates[-20:],
                'Actual Price': actual_prices[-20:],
                'Prediction': predictions[-20:],
                'True Value': y[-num_predictions:][-20:]
            })
            st.dataframe(recent_df, use_container_width=True)
            
        except Exception as e:
            st.error(f"❌ Error making predictions: {str(e)}")

# Tab 4: Evaluation
with tab4:
    st.header("Model Evaluation")
    
    if not st.session_state.model_trained:
        st.warning("⚠️ Please train a model first in the Train Model tab")
    else:
        try:
            trainer = st.session_state.trainer
            X = st.session_state.X
            y = st.session_state.y
            task = st.session_state.task
            
            # Make predictions
            predictions = trainer.predict(X)
            
            # Evaluate
            evaluator = ModelEvaluator()
            
            if task == 'regression':
                metrics = evaluator.evaluate_regression(y, predictions)
                
                st.subheader("Regression Metrics")
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("MSE", f"{metrics['mse']:.6f}")
                with col2:
                    st.metric("RMSE", f"{metrics['rmse']:.6f}")
                with col3:
                    st.metric("MAE", f"{metrics['mae']:.6f}")
                with col4:
                    st.metric("R²", f"{metrics['r2']:.6f}")
                
            else:
                metrics = evaluator.evaluate_classification(y, predictions)
                
                st.subheader("Classification Metrics")
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Accuracy", f"{metrics['accuracy']:.4f}")
                with col2:
                    st.metric("Precision", f"{metrics['precision']:.4f}")
                with col3:
                    st.metric("Recall", f"{metrics['recall']:.4f}")
                with col4:
                    st.metric("F1 Score", f"{metrics['f1']:.4f}")
                
                if 'confusion_matrix' in metrics:
                    st.subheader("Confusion Matrix")
                    cm = np.array(metrics['confusion_matrix'])
                    fig, ax = plt.subplots(figsize=(6, 5))
                    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
                    ax.figure.colorbar(im, ax=ax)
                    ax.set(xticks=np.arange(cm.shape[1]),
                           yticks=np.arange(cm.shape[0]),
                           xticklabels=['No-Buy', 'Buy'],
                           yticklabels=['No-Buy', 'Buy'],
                           title='Confusion Matrix',
                           ylabel='True label',
                           xlabel='Predicted label')
                    
                    # Add text annotations
                    for i in range(cm.shape[0]):
                        for j in range(cm.shape[1]):
                            ax.text(j, i, format(cm[i, j], 'd'),
                                   ha="center", va="center",
                                   color="white" if cm[i, j] > cm.max() / 2. else "black")
                    
                    st.pyplot(fig)
            
            # Detailed metrics table
            st.subheader("Detailed Metrics")
            metrics_df = pd.DataFrame([metrics])
            st.dataframe(metrics_df, use_container_width=True)
            
        except Exception as e:
            st.error(f"❌ Error evaluating model: {str(e)}")

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("### About")
st.sidebar.info(
    "This application uses deep learning (GRU/LSTM) to predict GLD price movements. "
    "It supports both regression (returns) and classification (buy/no-buy signals) "
    "at multiple time horizons (1, 5, 20 days)."
)
