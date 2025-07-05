"""
Data processor for high-frequency trading
Handles market data processing and feature extraction
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional
import asyncio
from concurrent.futures import ThreadPoolExecutor

from loguru import logger


class DataProcessor:
    """
    High-performance data processor
    Simplified version for initial setup
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.use_gpu = config.get('use_gpu', False)  # Start with CPU only
        
        # Data buffers
        self.price_buffer = []
        self.volume_buffer = []
        self.orderbook_buffer = []
        
        # Rolling windows
        self.window_sizes = [5, 10, 20, 50, 100]
        
        # Thread pool for parallel processing
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        logger.info("Data Processor initialized")
    
    async def process_market_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process incoming market data
        
        Args:
            data: Raw market data from exchange
            
        Returns:
            Processed data with indicators and features
        """
        try:
            data_type = data.get('type')
            
            if data_type == 'ticker':
                return await self._process_ticker_data(data)
            elif data_type == 'orderbook':
                return await self._process_orderbook_data(data)
            elif data_type == 'trade':
                return await self._process_trade_data(data)
            else:
                return data
                
        except Exception as e:
            logger.error(f"Error processing market data: {e}")
            return data
    
    async def _process_ticker_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process ticker data"""
        try:
            symbol = data.get('symbol')
            price = data.get('price', 0.0)
            volume = data.get('volume', 0.0)
            
            # Add to buffers
            self.price_buffer.append(price)
            self.volume_buffer.append(volume)
            
            # Keep buffer size manageable
            if len(self.price_buffer) > 1000:
                self.price_buffer = self.price_buffer[-1000:]
                self.volume_buffer = self.volume_buffer[-1000:]
            
            # Calculate basic indicators
            processed_data = data.copy()
            
            if len(self.price_buffer) >= 5:
                processed_data.update({
                    'sma_5': np.mean(self.price_buffer[-5:]),
                    'sma_10': np.mean(self.price_buffer[-10:]) if len(self.price_buffer) >= 10 else price,
                    'sma_20': np.mean(self.price_buffer[-20:]) if len(self.price_buffer) >= 20 else price,
                    'volatility': np.std(self.price_buffer[-20:]) if len(self.price_buffer) >= 20 else 0.0,
                    'rsi': self._calculate_rsi(self.price_buffer[-14:]) if len(self.price_buffer) >= 14 else 50.0,
                    'volume_avg': np.mean(self.volume_buffer[-20:]) if len(self.volume_buffer) >= 20 else volume
                })
            
            return processed_data
            
        except Exception as e:
            logger.error(f"Error processing ticker data: {e}")
            return data
    
    async def _process_orderbook_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process order book data"""
        try:
            bids = data.get('bids', [])
            asks = data.get('asks', [])
            
            if not bids or not asks:
                return data
            
            # Calculate order book metrics
            best_bid = bids[0][0] if bids else 0.0
            best_ask = asks[0][0] if asks else 0.0
            
            spread = best_ask - best_bid
            spread_pct = spread / best_bid if best_bid > 0 else 0.0
            mid_price = (best_bid + best_ask) / 2
            
            # Order book imbalance
            bid_volume = sum(bid[1] for bid in bids[:5])  # Top 5 levels
            ask_volume = sum(ask[1] for ask in asks[:5])  # Top 5 levels
            
            total_volume = bid_volume + ask_volume
            imbalance = (bid_volume - ask_volume) / total_volume if total_volume > 0 else 0.0
            
            # Add calculated metrics
            processed_data = data.copy()
            processed_data.update({
                'best_bid': best_bid,
                'best_ask': best_ask,
                'spread': spread,
                'spread_pct': spread_pct,
                'mid_price': mid_price,
                'bid_volume': bid_volume,
                'ask_volume': ask_volume,
                'imbalance': imbalance
            })
            
            return processed_data
            
        except Exception as e:
            logger.error(f"Error processing orderbook data: {e}")
            return data
    
    async def _process_trade_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process trade data"""
        try:
            # Add basic trade metrics
            processed_data = data.copy()
            
            # Trade intensity (could be enhanced with more data)
            processed_data.update({
                'trade_intensity': 1.0,  # Placeholder
                'price_impact': 0.0     # Placeholder
            })
            
            return processed_data
            
        except Exception as e:
            logger.error(f"Error processing trade data: {e}")
            return data
    
    def _calculate_rsi(self, prices: List[float], period: int = 14) -> float:
        """Calculate RSI indicator"""
        try:
            if len(prices) < period:
                return 50.0
            
            prices_array = np.array(prices)
            deltas = np.diff(prices_array)
            
            gains = np.where(deltas > 0, deltas, 0)
            losses = np.where(deltas < 0, -deltas, 0)
            
            avg_gain = np.mean(gains[-period:])
            avg_loss = np.mean(losses[-period:])
            
            if avg_loss == 0:
                return 100.0
            
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
            
            return rsi
            
        except Exception as e:
            logger.error(f"Error calculating RSI: {e}")
            return 50.0
    
    def _calculate_bollinger_bands(self, prices: List[float], period: int = 20, std_dev: int = 2) -> Dict[str, float]:
        """Calculate Bollinger Bands"""
        try:
            if len(prices) < period:
                current_price = prices[-1] if prices else 0.0
                return {
                    'bb_upper': current_price,
                    'bb_middle': current_price,
                    'bb_lower': current_price
                }
            
            prices_array = np.array(prices[-period:])
            middle = np.mean(prices_array)
            std = np.std(prices_array)
            
            upper = middle + (std * std_dev)
            lower = middle - (std * std_dev)
            
            return {
                'bb_upper': upper,
                'bb_middle': middle,
                'bb_lower': lower
            }
            
        except Exception as e:
            logger.error(f"Error calculating Bollinger Bands: {e}")
            current_price = prices[-1] if prices else 0.0
            return {
                'bb_upper': current_price,
                'bb_middle': current_price,
                'bb_lower': current_price
            }
    
    async def calculate_features(self, data: Dict[str, Any]) -> np.ndarray:
        """
        Calculate ML features from processed data
        
        Args:
            data: Processed market data
            
        Returns:
            Feature vector for ML models
        """
        try:
            features = []
            
            # Price features
            features.extend([
                data.get('price', 0.0),
                data.get('sma_5', 0.0),
                data.get('sma_10', 0.0),
                data.get('sma_20', 0.0),
                data.get('volatility', 0.0),
                data.get('rsi', 50.0)
            ])
            
            # Volume features
            features.extend([
                data.get('volume', 0.0),
                data.get('volume_avg', 0.0)
            ])
            
            # Order book features (if available)
            features.extend([
                data.get('spread_pct', 0.0),
                data.get('imbalance', 0.0),
                data.get('bid_volume', 0.0),
                data.get('ask_volume', 0.0)
            ])
            
            return np.array(features, dtype=np.float32)
            
        except Exception as e:
            logger.error(f"Error calculating features: {e}")
            return np.zeros(12, dtype=np.float32)  # Return default feature vector
    
    def get_market_state(self) -> Dict[str, Any]:
        """Get current market state"""
        try:
            if not self.price_buffer:
                return {'state': 'no_data'}
            
            current_price = self.price_buffer[-1]
            
            # Determine market trend
            if len(self.price_buffer) >= 20:
                sma_short = np.mean(self.price_buffer[-5:])
                sma_long = np.mean(self.price_buffer[-20:])
                
                if sma_short > sma_long * 1.001:
                    trend = 'bullish'
                elif sma_short < sma_long * 0.999:
                    trend = 'bearish'
                else:
                    trend = 'sideways'
            else:
                trend = 'unknown'
            
            # Calculate volatility
            volatility = np.std(self.price_buffer[-20:]) if len(self.price_buffer) >= 20 else 0.0
            
            return {
                'state': 'active',
                'current_price': current_price,
                'trend': trend,
                'volatility': volatility,
                'data_points': len(self.price_buffer)
            }
            
        except Exception as e:
            logger.error(f"Error getting market state: {e}")
            return {'state': 'error'}
    
    def reset_buffers(self):
        """Reset all data buffers"""
        self.price_buffer.clear()
        self.volume_buffer.clear()
        self.orderbook_buffer.clear()
        logger.info("Data buffers reset")
    
    def get_buffer_info(self) -> Dict[str, int]:
        """Get buffer information"""
        return {
            'price_buffer_size': len(self.price_buffer),
            'volume_buffer_size': len(self.volume_buffer),
            'orderbook_buffer_size': len(self.orderbook_buffer)
        } 