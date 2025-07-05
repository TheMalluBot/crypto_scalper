"""
Machine Learning based trading strategies
Uses AI models for price prediction and signal generation
"""

import time
from typing import Dict, List, Optional, Any
import numpy as np
from loguru import logger

from .base_strategy import BaseStrategy
from ..core.order_manager import TradeSignal, OrderSide


class MLPredictionStrategy(BaseStrategy):
    """
    Machine Learning prediction strategy
    Uses trained models to predict price movements
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__("MLPrediction", config)
        
        # Strategy parameters
        self.model_type = config.get('model_type', 'LSTM')
        self.prediction_horizon = config.get('prediction_horizon', 60)  # seconds
        self.confidence_threshold = config.get('confidence_threshold', 0.75)
        self.retrain_interval = config.get('retrain_interval', 3600)  # 1 hour
        
        # Feature parameters
        self.lookback_window = 100
        self.feature_dim = 12
        
        # Model state
        self.model = None
        self.scaler = None
        self.last_retrain_time = 0.0
        
        # Feature buffers
        self.feature_buffer = []
        self.price_buffer = []
        self.volume_buffer = []
        
        # Prediction tracking
        self.predictions_history = []
        self.model_accuracy = 0.0
        
        logger.info(f"ML Prediction strategy initialized with {self.model_type} model")
    
    async def generate_signal(self) -> Optional[TradeSignal]:
        """Generate ML-based trading signals"""
        try:
            if not self.active or not self.current_market_data:
                return None
            
            # Check if we have enough data
            if len(self.feature_buffer) < self.lookback_window:
                return None
            
            # Initialize or retrain model if needed
            if self.model is None or await self._should_retrain():
                await self._initialize_model()
            
            # Generate prediction
            prediction = await self._predict_price_movement()
            
            if prediction is None:
                return None
            
            # Convert prediction to trading signal
            signal = await self._prediction_to_signal(prediction)
            
            return signal
            
        except Exception as e:
            logger.error(f"Error generating ML prediction signal: {e}")
            return None
    
    async def update_market_data(self, data: Dict[str, Any]):
        """Update market data and features for ML model"""
        try:
            self.current_market_data = data
            
            # Extract features from market data
            features = await self._extract_features(data)
            
            if features is not None:
                self.feature_buffer.append(features)
                
                # Keep buffer size manageable
                if len(self.feature_buffer) > 1000:
                    self.feature_buffer = self.feature_buffer[-1000:]
            
            # Track prices for training data
            if 'price' in data:
                self.price_buffer.append(data['price'])
                if len(self.price_buffer) > 1000:
                    self.price_buffer = self.price_buffer[-1000:]
            
            # Track volume
            if 'volume' in data:
                self.volume_buffer.append(data['volume'])
                if len(self.volume_buffer) > 1000:
                    self.volume_buffer = self.volume_buffer[-1000:]
            
        except Exception as e:
            logger.error(f"Error updating ML market data: {e}")
    
    async def _extract_features(self, data: Dict[str, Any]) -> Optional[np.ndarray]:
        """Extract features from market data for ML model"""
        try:
            features = []
            
            # Price features
            price = data.get('price', 0.0)
            features.append(price)
            
            # Technical indicators
            features.extend([
                data.get('sma_5', price),
                data.get('sma_10', price),
                data.get('sma_20', price),
                data.get('rsi', 50.0),
                data.get('volatility', 0.0)
            ])
            
            # Volume features
            volume = data.get('volume', 0.0)
            features.extend([
                volume,
                data.get('volume_avg', volume)
            ])
            
            # Order book features
            features.extend([
                data.get('spread_pct', 0.0),
                data.get('imbalance', 0.0),
                data.get('bid_volume', 0.0),
                data.get('ask_volume', 0.0)
            ])
            
            # Normalize price by last known price if available
            if len(self.price_buffer) > 0:
                last_price = self.price_buffer[-1]
                if last_price > 0:
                    features[0] = price / last_price - 1  # Price change ratio
            
            return np.array(features, dtype=np.float32)
            
        except Exception as e:
            logger.error(f"Error extracting features: {e}")
            return None
    
    async def _initialize_model(self):
        """Initialize or retrain the ML model"""
        try:
            logger.info("Initializing ML model...")
            
            # Simple placeholder model (Linear regression style)
            # In production, this would load a proper trained model
            self.model = SimplePredictor()
            
            # Train on historical data if available
            if len(self.feature_buffer) >= self.lookback_window and len(self.price_buffer) >= self.lookback_window:
                await self._train_model()
            
            self.last_retrain_time = time.time()
            logger.info("ML model initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing ML model: {e}")
    
    async def _train_model(self):
        """Train the ML model with historical data"""
        try:
            if len(self.feature_buffer) < self.lookback_window:
                return
            
            # Prepare training data
            X, y = self._prepare_training_data()
            
            if X is None or y is None:
                return
            
            # Train the model
            self.model.fit(X, y)
            
            # Calculate model accuracy on recent data
            if len(X) > 20:
                recent_X = X[-20:]
                recent_y = y[-20:]
                predictions = self.model.predict(recent_X)
                
                # Calculate accuracy as percentage of correct direction predictions
                correct_predictions = 0
                for i in range(len(predictions)):
                    if (predictions[i] > 0 and recent_y[i] > 0) or (predictions[i] < 0 and recent_y[i] < 0):
                        correct_predictions += 1
                
                self.model_accuracy = correct_predictions / len(predictions)
                logger.info(f"Model retrained. Accuracy: {self.model_accuracy:.2%}")
            
        except Exception as e:
            logger.error(f"Error training ML model: {e}")
    
    def _prepare_training_data(self) -> tuple:
        """Prepare training data from historical features and prices"""
        try:
            if len(self.feature_buffer) < self.lookback_window or len(self.price_buffer) < self.lookback_window:
                return None, None
            
            X = []
            y = []
            
            # Create sequences for training
            for i in range(self.lookback_window, len(self.feature_buffer) - 1):
                # Features: last N observations
                features = self.feature_buffer[i-self.lookback_window:i]
                X.append(np.array(features).flatten())
                
                # Target: future price change
                current_price = self.price_buffer[i]
                future_price = self.price_buffer[i + 1]
                
                if current_price > 0:
                    price_change = (future_price - current_price) / current_price
                    y.append(price_change)
                else:
                    y.append(0.0)
            
            return np.array(X), np.array(y)
            
        except Exception as e:
            logger.error(f"Error preparing training data: {e}")
            return None, None
    
    async def _predict_price_movement(self) -> Optional[Dict[str, float]]:
        """Predict future price movement using the ML model"""
        try:
            if self.model is None or len(self.feature_buffer) < self.lookback_window:
                return None
            
            # Prepare input features
            recent_features = self.feature_buffer[-self.lookback_window:]
            X = np.array(recent_features).flatten().reshape(1, -1)
            
            # Make prediction
            prediction = self.model.predict(X)[0]
            confidence = self.model.get_confidence(X)[0] if hasattr(self.model, 'get_confidence') else abs(prediction)
            
            # Store prediction in history
            self.predictions_history.append({
                'prediction': prediction,
                'confidence': confidence,
                'timestamp': time.time()
            })
            
            # Keep history manageable
            if len(self.predictions_history) > 100:
                self.predictions_history = self.predictions_history[-100:]
            
            return {
                'price_change': prediction,
                'confidence': confidence,
                'direction': 1 if prediction > 0 else -1
            }
            
        except Exception as e:
            logger.error(f"Error predicting price movement: {e}")
            return None
    
    async def _prediction_to_signal(self, prediction: Dict[str, float]) -> Optional[TradeSignal]:
        """Convert ML prediction to trading signal"""
        try:
            price_change = prediction['price_change']
            confidence = prediction['confidence']
            direction = prediction['direction']
            
            # Check confidence threshold
            if confidence < self.confidence_threshold:
                return None
            
            # Determine signal strength based on prediction magnitude and confidence
            signal_strength = min(abs(price_change) * 100 + confidence * 0.5, 1.0)
            
            # Only generate signals for significant predictions
            if abs(price_change) < 0.001:  # Less than 0.1% predicted change
                return None
            
            # Determine trade direction
            if direction > 0:
                side = OrderSide.BUY
            else:
                side = OrderSide.SELL
            
            # Get current price
            current_price = self.current_market_data.get('price', 0.0)
            if current_price <= 0:
                return None
            
            # Calculate risk levels based on prediction
            predicted_price = current_price * (1 + price_change)
            
            # Conservative risk management
            stop_loss_pct = 0.003  # 0.3% stop loss
            take_profit_pct = min(abs(price_change) * 0.7, 0.01)  # Max 1% take profit
            
            if side == OrderSide.BUY:
                stop_loss = current_price * (1 - stop_loss_pct)
                take_profit = current_price * (1 + take_profit_pct)
            else:
                stop_loss = current_price * (1 + stop_loss_pct)
                take_profit = current_price * (1 - take_profit_pct)
            
            signal = TradeSignal(
                symbol=self.current_market_data.get('symbol', 'BTCUSDT'),
                side=side,
                signal_strength=signal_strength,
                price_target=current_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                urgency="NORMAL",
                strategy_id=self.name,
                timestamp=time.time()
            )
            
            return signal
            
        except Exception as e:
            logger.error(f"Error converting prediction to signal: {e}")
            return None
    
    async def _should_retrain(self) -> bool:
        """Check if model should be retrained"""
        current_time = time.time()
        
        # Time-based retraining
        if current_time - self.last_retrain_time > self.retrain_interval:
            return True
        
        # Performance-based retraining
        if self.model_accuracy < 0.55:  # Below 55% accuracy
            return True
        
        return False
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the ML model"""
        return {
            'model_type': self.model_type,
            'model_accuracy': self.model_accuracy,
            'last_retrain_time': self.last_retrain_time,
            'predictions_count': len(self.predictions_history),
            'feature_buffer_size': len(self.feature_buffer),
            'confidence_threshold': self.confidence_threshold
        }


class SimplePredictor:
    """
    Simple predictor model for demonstration
    In production, this would be replaced with proper ML models
    """
    
    def __init__(self):
        self.weights = None
        self.bias = 0.0
        self.is_trained = False
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        """Fit the model using simple linear regression"""
        try:
            if len(X) == 0 or len(y) == 0:
                return
            
            # Simple linear regression
            X_mean = np.mean(X, axis=0)
            y_mean = np.mean(y)
            
            # Calculate weights (simplified)
            numerator = np.sum((X - X_mean) * (y - y_mean).reshape(-1, 1), axis=0)
            denominator = np.sum((X - X_mean) ** 2, axis=0)
            
            # Avoid division by zero
            denominator = np.where(denominator == 0, 1, denominator)
            self.weights = numerator / denominator
            self.bias = y_mean - np.dot(X_mean, self.weights)
            
            self.is_trained = True
            
        except Exception as e:
            logger.error(f"Error fitting simple predictor: {e}")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions"""
        try:
            if not self.is_trained or self.weights is None:
                return np.zeros(len(X))
            
            return np.dot(X, self.weights) + self.bias
            
        except Exception as e:
            logger.error(f"Error making prediction: {e}")
            return np.zeros(len(X))
    
    def get_confidence(self, X: np.ndarray) -> np.ndarray:
        """Get prediction confidence (simplified)"""
        try:
            predictions = self.predict(X)
            # Simple confidence based on prediction magnitude
            return np.clip(np.abs(predictions) * 10, 0.1, 1.0)
            
        except Exception as e:
            logger.error(f"Error calculating confidence: {e}")
            return np.full(len(X), 0.5) 