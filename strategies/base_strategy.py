"""
Base Strategy class for crypto scalping
All trading strategies inherit from this class
"""

import asyncio
import time
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum

import numpy as np
from loguru import logger

from ..core.order_manager import TradeSignal, OrderSide


class SignalStrength(Enum):
    WEAK = 0.3
    MEDIUM = 0.6
    STRONG = 0.8
    VERY_STRONG = 1.0


@dataclass
class StrategyMetrics:
    """Strategy performance metrics"""
    total_signals: int = 0
    successful_signals: int = 0
    false_signals: int = 0
    avg_signal_strength: float = 0.0
    avg_hold_time: float = 0.0
    total_pnl: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    win_rate: float = 0.0


class BaseStrategy(ABC):
    """
    Abstract base class for all trading strategies
    Provides common functionality and interface
    """
    
    def __init__(self, name: str, config: Dict[str, Any]):
        self.name = name
        self.config = config
        self.enabled = config.get('enabled', True)
        
        # Strategy state
        self.active = False
        self.last_signal_time = 0.0
        self.signal_cooldown = config.get('signal_cooldown', 1.0)  # seconds
        
        # Performance tracking
        self.metrics = StrategyMetrics()
        self.signal_history = []
        
        # Market data
        self.current_market_data = {}
        self.price_history = []
        self.volume_history = []
        
        # Strategy parameters
        self.min_signal_strength = config.get('min_signal_strength', 0.5)
        self.max_position_size = config.get('max_position_size', 0.02)
        
        logger.info(f"Strategy '{name}' initialized")
    
    @abstractmethod
    async def generate_signal(self) -> Optional[TradeSignal]:
        """
        Generate trading signal based on current market conditions
        
        Returns:
            TradeSignal if conditions are met, None otherwise
        """
        pass
    
    @abstractmethod
    async def update_market_data(self, data: Dict[str, Any]):
        """
        Update strategy with new market data
        
        Args:
            data: Market data from data processor
        """
        pass
    
    async def process_signal(self, signal: TradeSignal) -> bool:
        """
        Process and validate a trading signal
        
        Args:
            signal: Generated trading signal
            
        Returns:
            True if signal is valid and should be executed
        """
        try:
            # Check cooldown period
            if not await self._check_signal_cooldown():
                return False
            
            # Validate signal strength
            if signal.signal_strength < self.min_signal_strength:
                logger.debug(f"Signal strength too low: {signal.signal_strength}")
                return False
            
            # Validate signal parameters
            if not await self._validate_signal(signal):
                return False
            
            # Update metrics
            self.metrics.total_signals += 1
            self.last_signal_time = time.time()
            
            # Store signal in history
            self.signal_history.append({
                'signal': signal,
                'timestamp': time.time(),
                'executed': True
            })
            
            logger.info(f"Signal processed: {signal.symbol} {signal.side.value} "
                       f"strength={signal.signal_strength:.2f}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error processing signal: {e}")
            return False
    
    async def update_performance(self, trade_result: Dict[str, Any]):
        """
        Update strategy performance metrics
        
        Args:
            trade_result: Result of executed trade
        """
        try:
            pnl = trade_result.get('pnl', 0.0)
            self.metrics.total_pnl += pnl
            
            if pnl > 0:
                self.metrics.successful_signals += 1
            else:
                self.metrics.false_signals += 1
            
            # Update win rate
            total_trades = self.metrics.successful_signals + self.metrics.false_signals
            if total_trades > 0:
                self.metrics.win_rate = self.metrics.successful_signals / total_trades
            
            # Update average signal strength
            if len(self.signal_history) > 0:
                strengths = [s['signal'].signal_strength for s in self.signal_history[-100:]]
                self.metrics.avg_signal_strength = np.mean(strengths)
            
        except Exception as e:
            logger.error(f"Error updating performance: {e}")
    
    async def _check_signal_cooldown(self) -> bool:
        """Check if enough time has passed since last signal"""
        current_time = time.time()
        return (current_time - self.last_signal_time) >= self.signal_cooldown
    
    async def _validate_signal(self, signal: TradeSignal) -> bool:
        """
        Validate signal parameters
        
        Args:
            signal: Trading signal to validate
            
        Returns:
            True if signal is valid
        """
        try:
            # Check required fields
            if not signal.symbol or not signal.side:
                return False
            
            # Check signal strength bounds
            if not (0.0 <= signal.signal_strength <= 1.0):
                return False
            
            # Check price levels
            if signal.price_target and signal.price_target <= 0:
                return False
            
            if signal.stop_loss and signal.stop_loss <= 0:
                return False
            
            if signal.take_profit and signal.take_profit <= 0:
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error validating signal: {e}")
            return False
    
    def calculate_signal_strength(self, indicators: Dict[str, float]) -> float:
        """
        Calculate signal strength based on technical indicators
        
        Args:
            indicators: Technical indicator values
            
        Returns:
            Signal strength between 0.0 and 1.0
        """
        try:
            # Base implementation - strategies can override
            strength = 0.5  # Neutral strength
            
            # Example: Use RSI to adjust strength
            rsi = indicators.get('rsi', 50.0)
            if rsi < 30:  # Oversold
                strength += 0.2
            elif rsi > 70:  # Overbought
                strength += 0.2
            
            # Example: Use trend indicators
            sma_5 = indicators.get('sma_5', 0.0)
            sma_20 = indicators.get('sma_20', 0.0)
            if sma_5 > sma_20:  # Uptrend
                strength += 0.1
            elif sma_5 < sma_20:  # Downtrend
                strength += 0.1
            
            # Cap strength at 1.0
            return min(strength, 1.0)
            
        except Exception as e:
            logger.error(f"Error calculating signal strength: {e}")
            return 0.5
    
    def get_risk_levels(self, current_price: float) -> Dict[str, float]:
        """
        Calculate stop loss and take profit levels
        
        Args:
            current_price: Current market price
            
        Returns:
            Dictionary with stop_loss and take_profit levels
        """
        try:
            stop_loss_pct = self.config.get('stop_loss', 0.005)  # 0.5%
            take_profit_pct = self.config.get('profit_target', 0.015)  # 1.5%
            
            return {
                'stop_loss': current_price * (1 - stop_loss_pct),
                'take_profit': current_price * (1 + take_profit_pct)
            }
            
        except Exception as e:
            logger.error(f"Error calculating risk levels: {e}")
            return {'stop_loss': current_price * 0.995, 'take_profit': current_price * 1.015}
    
    def is_market_condition_suitable(self, market_data: Dict[str, Any]) -> bool:
        """
        Check if current market conditions are suitable for this strategy
        
        Args:
            market_data: Current market data
            
        Returns:
            True if conditions are suitable
        """
        try:
            # Check spread
            spread_pct = market_data.get('spread_pct', 0.0)
            max_spread = self.config.get('max_spread_pct', 0.002)
            if spread_pct > max_spread:
                return False
            
            # Check volume
            volume = market_data.get('volume', 0.0)
            min_volume = self.config.get('min_volume', 1000.0)
            if volume < min_volume:
                return False
            
            # Check volatility
            volatility = market_data.get('volatility', 0.0)
            max_volatility = self.config.get('max_volatility', 0.05)
            if volatility > max_volatility:
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error checking market conditions: {e}")
            return False
    
    def activate(self):
        """Activate the strategy"""
        self.active = True
        logger.info(f"Strategy '{self.name}' activated")
    
    def deactivate(self):
        """Deactivate the strategy"""
        self.active = False
        logger.info(f"Strategy '{self.name}' deactivated")
    
    def reset_metrics(self):
        """Reset performance metrics"""
        self.metrics = StrategyMetrics()
        self.signal_history.clear()
        logger.info(f"Strategy '{self.name}' metrics reset")
    
    def get_metrics(self) -> StrategyMetrics:
        """Get current performance metrics"""
        return self.metrics
    
    def get_status(self) -> Dict[str, Any]:
        """Get strategy status"""
        return {
            'name': self.name,
            'active': self.active,
            'enabled': self.enabled,
            'total_signals': self.metrics.total_signals,
            'win_rate': self.metrics.win_rate,
            'total_pnl': self.metrics.total_pnl,
            'avg_signal_strength': self.metrics.avg_signal_strength,
            'last_signal_time': self.last_signal_time
        }
    
    def get_recent_signals(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent signals"""
        return self.signal_history[-limit:] if self.signal_history else [] 