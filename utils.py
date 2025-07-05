"""
Utility functions for the crypto scalping project
Common helpers and tools used across the system
"""

import asyncio
import time
import json
import yaml
from pathlib import Path
from typing import Dict, List, Any, Optional
import numpy as np
from loguru import logger


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file"""
    try:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        logger.info(f"Configuration loaded from {config_path}")
        return config
    except Exception as e:
        logger.error(f"Error loading config from {config_path}: {e}")
        return {}


def save_config(config: Dict[str, Any], config_path: str):
    """Save configuration to YAML file"""
    try:
        with open(config_path, 'w') as file:
            yaml.dump(config, file, default_flow_style=False)
        logger.info(f"Configuration saved to {config_path}")
    except Exception as e:
        logger.error(f"Error saving config to {config_path}: {e}")


def validate_api_keys(config: Dict[str, Any]) -> bool:
    """Validate that API keys are present in configuration"""
    try:
        binance_config = config.get('binance', {})
        
        api_key = binance_config.get('api_key', '')
        api_secret = binance_config.get('api_secret', '')
        
        if not api_key or api_key == 'your_api_key_here':
            logger.warning("Binance API key not configured")
            return False
        
        if not api_secret or api_secret == 'your_api_secret_here':
            logger.warning("Binance API secret not configured")
            return False
        
        return True
        
    except Exception as e:
        logger.error(f"Error validating API keys: {e}")
        return False


def calculate_position_size(portfolio_value: float, risk_per_trade: float, 
                          stop_loss_pct: float, price: float) -> float:
    """
    Calculate position size based on Kelly Criterion and risk management
    
    Args:
        portfolio_value: Total portfolio value
        risk_per_trade: Risk per trade as percentage (e.g., 0.02 for 2%)
        stop_loss_pct: Stop loss as percentage (e.g., 0.01 for 1%)
        price: Current price of the asset
        
    Returns:
        Position size in base currency
    """
    try:
        risk_amount = portfolio_value * risk_per_trade
        
        if stop_loss_pct <= 0:
            logger.warning("Invalid stop loss percentage")
            return 0.0
        
        # Calculate shares/units that can be bought with risk amount
        position_size = risk_amount / (price * stop_loss_pct)
        
        return max(0.0, position_size)
        
    except Exception as e:
        logger.error(f"Error calculating position size: {e}")
        return 0.0


def format_currency(amount: float, currency: str = 'USD') -> str:
    """Format currency amount for display"""
    try:
        if currency == 'USD':
            return f"${amount:,.2f}"
        else:
            return f"{amount:,.6f} {currency}"
    except:
        return f"{amount} {currency}"


def format_percentage(value: float) -> str:
    """Format percentage for display"""
    try:
        return f"{value * 100:.2f}%"
    except:
        return f"{value}%"


def get_timestamp() -> int:
    """Get current timestamp in milliseconds"""
    return int(time.time() * 1000)


def time_to_string(timestamp: float) -> str:
    """Convert timestamp to readable string"""
    try:
        import datetime
        dt = datetime.datetime.fromtimestamp(timestamp)
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    except:
        return str(timestamp)


def calculate_returns(prices: List[float]) -> List[float]:
    """Calculate returns from price series"""
    try:
        if len(prices) < 2:
            return []
        
        returns = []
        for i in range(1, len(prices)):
            if prices[i-1] > 0:
                ret = (prices[i] - prices[i-1]) / prices[i-1]
                returns.append(ret)
            else:
                returns.append(0.0)
        
        return returns
        
    except Exception as e:
        logger.error(f"Error calculating returns: {e}")
        return []


def calculate_sharpe_ratio(returns: List[float], risk_free_rate: float = 0.02) -> float:
    """Calculate Sharpe ratio"""
    try:
        if len(returns) < 2:
            return 0.0
        
        returns_array = np.array(returns)
        
        # Annualized return
        avg_return = np.mean(returns_array) * 252  # Assuming daily returns
        
        # Annualized volatility
        volatility = np.std(returns_array) * np.sqrt(252)
        
        if volatility == 0:
            return 0.0
        
        sharpe = (avg_return - risk_free_rate) / volatility
        return sharpe
        
    except Exception as e:
        logger.error(f"Error calculating Sharpe ratio: {e}")
        return 0.0


def calculate_max_drawdown(equity_curve: List[float]) -> float:
    """Calculate maximum drawdown from equity curve"""
    try:
        if len(equity_curve) < 2:
            return 0.0
        
        peak = equity_curve[0]
        max_dd = 0.0
        
        for value in equity_curve[1:]:
            if value > peak:
                peak = value
            
            drawdown = (peak - value) / peak
            max_dd = max(max_dd, drawdown)
        
        return max_dd
        
    except Exception as e:
        logger.error(f"Error calculating max drawdown: {e}")
        return 0.0


def calculate_volatility(returns: List[float], periods: int = 252) -> float:
    """Calculate annualized volatility"""
    try:
        if len(returns) < 2:
            return 0.0
        
        returns_array = np.array(returns)
        volatility = np.std(returns_array) * np.sqrt(periods)
        
        return volatility
        
    except Exception as e:
        logger.error(f"Error calculating volatility: {e}")
        return 0.0


def kelly_criterion(win_rate: float, avg_win: float, avg_loss: float) -> float:
    """
    Calculate Kelly Criterion for optimal position sizing
    
    Args:
        win_rate: Win rate (0.0 to 1.0)
        avg_win: Average winning trade
        avg_loss: Average losing trade (positive value)
        
    Returns:
        Kelly percentage (0.0 to 1.0)
    """
    try:
        if avg_loss <= 0 or win_rate <= 0 or win_rate >= 1:
            return 0.0
        
        loss_rate = 1 - win_rate
        
        # Kelly formula: f = (bp - q) / b
        # where b = avg_win/avg_loss, p = win_rate, q = loss_rate
        b = avg_win / avg_loss
        kelly = (b * win_rate - loss_rate) / b
        
        # Cap Kelly at reasonable levels (max 25%)
        kelly = max(0.0, min(kelly, 0.25))
        
        return kelly
        
    except Exception as e:
        logger.error(f"Error calculating Kelly criterion: {e}")
        return 0.0


def round_to_precision(value: float, precision: int) -> float:
    """Round value to specified decimal precision"""
    try:
        return round(value, precision)
    except:
        return value


def truncate_to_precision(value: float, precision: int) -> float:
    """Truncate value to specified decimal precision"""
    try:
        multiplier = 10 ** precision
        return int(value * multiplier) / multiplier
    except:
        return value


def create_directory(path: str):
    """Create directory if it doesn't exist"""
    try:
        Path(path).mkdir(parents=True, exist_ok=True)
    except Exception as e:
        logger.error(f"Error creating directory {path}: {e}")


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Safe division with default value for zero denominator"""
    try:
        if denominator == 0:
            return default
        return numerator / denominator
    except:
        return default


def clamp(value: float, min_val: float, max_val: float) -> float:
    """Clamp value between min and max"""
    return max(min_val, min(value, max_val))


def exponential_backoff(attempt: int, base_delay: float = 1.0, max_delay: float = 60.0) -> float:
    """Calculate exponential backoff delay"""
    delay = base_delay * (2 ** attempt)
    return min(delay, max_delay)


async def retry_async(func, max_attempts: int = 3, base_delay: float = 1.0):
    """Retry async function with exponential backoff"""
    for attempt in range(max_attempts):
        try:
            return await func()
        except Exception as e:
            if attempt == max_attempts - 1:
                raise e
            
            delay = exponential_backoff(attempt, base_delay)
            logger.warning(f"Attempt {attempt + 1} failed, retrying in {delay:.1f}s: {e}")
            await asyncio.sleep(delay)


def validate_symbol(symbol: str) -> bool:
    """Validate trading symbol format"""
    try:
        if not symbol or len(symbol) < 6:
            return False
        
        # Basic validation for crypto pairs
        if not symbol.isalpha():
            return False
        
        # Common crypto pair endings
        valid_endings = ['USDT', 'BTC', 'ETH', 'BNB', 'BUSD']
        
        for ending in valid_endings:
            if symbol.endswith(ending):
                return True
        
        return False
        
    except:
        return False


def validate_price(price: float, min_price: float = 0.0) -> bool:
    """Validate price value"""
    try:
        return price > min_price and np.isfinite(price)
    except:
        return False


def validate_quantity(quantity: float, min_qty: float = 0.0) -> bool:
    """Validate quantity value"""
    try:
        return quantity > min_qty and np.isfinite(quantity)
    except:
        return False


def get_lot_size(symbol: str) -> Dict[str, float]:
    """Get lot size information for symbol (placeholder)"""
    # This would normally fetch from exchange info
    return {
        'min_qty': 0.00001,
        'max_qty': 9000000.0,
        'step_size': 0.00001
    }


def adjust_quantity_to_lot_size(quantity: float, lot_info: Dict[str, float]) -> float:
    """Adjust quantity to comply with lot size rules"""
    try:
        min_qty = lot_info.get('min_qty', 0.0)
        max_qty = lot_info.get('max_qty', float('inf'))
        step_size = lot_info.get('step_size', 0.00001)
        
        # Clamp to min/max
        quantity = clamp(quantity, min_qty, max_qty)
        
        # Adjust to step size
        if step_size > 0:
            quantity = truncate_to_precision(quantity, len(str(step_size).split('.')[-1]))
        
        return max(quantity, min_qty)
        
    except Exception as e:
        logger.error(f"Error adjusting quantity to lot size: {e}")
        return quantity


def calculate_fees(quantity: float, price: float, fee_rate: float = 0.001) -> float:
    """Calculate trading fees"""
    try:
        return quantity * price * fee_rate
    except:
        return 0.0


def save_performance_data(data: Dict[str, Any], filename: str):
    """Save performance data to JSON file"""
    try:
        with open(filename, 'w') as file:
            json.dump(data, file, indent=2, default=str)
        logger.info(f"Performance data saved to {filename}")
    except Exception as e:
        logger.error(f"Error saving performance data: {e}")


def load_performance_data(filename: str) -> Dict[str, Any]:
    """Load performance data from JSON file"""
    try:
        with open(filename, 'r') as file:
            data = json.load(file)
        logger.info(f"Performance data loaded from {filename}")
        return data
    except Exception as e:
        logger.error(f"Error loading performance data: {e}")
        return {}


class RateLimiter:
    """Simple rate limiter for API calls"""
    
    def __init__(self, max_calls: int, time_window: float):
        self.max_calls = max_calls
        self.time_window = time_window
        self.calls = []
    
    async def acquire(self):
        """Acquire rate limit token"""
        current_time = time.time()
        
        # Remove old calls outside time window
        self.calls = [call_time for call_time in self.calls 
                     if current_time - call_time < self.time_window]
        
        # Check if we can make a call
        if len(self.calls) >= self.max_calls:
            sleep_time = self.time_window - (current_time - self.calls[0])
            if sleep_time > 0:
                await asyncio.sleep(sleep_time)
                return await self.acquire()
        
        # Record this call
        self.calls.append(current_time)


class PerformanceTimer:
    """Context manager for timing operations"""
    
    def __init__(self, operation_name: str):
        self.operation_name = operation_name
        self.start_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.start_time:
            duration = time.time() - self.start_time
            logger.debug(f"{self.operation_name} took {duration:.3f}s") 