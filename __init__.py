"""
Crypto Scalper - High-Frequency Trading System for Cryptocurrency
Professional-grade algorithmic scalping platform with GPU acceleration
"""

__version__ = "1.0.0"
__author__ = "Crypto Scalper Team"
__description__ = "High-frequency crypto trading system with advanced strategies"

from .main import ScalpingApp
from .core.trading_engine import TradingEngine
from .core.order_manager import OrderManager
from .core.position_manager import PositionManager
from .core.risk_manager import RiskManager

from .data.binance_client import BinanceClient
from .data.data_processor import DataProcessor

from .strategies.base_strategy import BaseStrategy
from .strategies.scalping_strategies import MicroScalpingStrategy, OrderBookImbalanceStrategy
from .strategies.ml_strategies import MLPredictionStrategy

from .utils import (
    load_config,
    calculate_position_size,
    kelly_criterion,
    calculate_sharpe_ratio,
    calculate_max_drawdown,
    format_currency,
    format_percentage,
    RateLimiter,
    PerformanceTimer
)

__all__ = [
    # Main application
    "ScalpingApp",
    
    # Core components
    "TradingEngine",
    "OrderManager", 
    "PositionManager",
    "RiskManager",
    
    # Data components
    "BinanceClient",
    "DataProcessor",
    
    # Strategy components
    "BaseStrategy",
    "MicroScalpingStrategy",
    "OrderBookImbalanceStrategy", 
    "MLPredictionStrategy",
    
    # Utility functions
    "load_config",
    "calculate_position_size",
    "kelly_criterion",
    "calculate_sharpe_ratio",
    "calculate_max_drawdown",
    "format_currency",
    "format_percentage",
    "RateLimiter",
    "PerformanceTimer"
] 