"""
Scalping strategies for high-frequency crypto trading
Implements various scalping approaches
"""

import time
from typing import Dict, List, Optional, Any
import numpy as np
from loguru import logger

from .base_strategy import BaseStrategy
from ..core.order_manager import TradeSignal, OrderSide


class MicroScalpingStrategy(BaseStrategy):
    """
    Micro scalping strategy for very short-term trades
    Targets small price movements with tight risk management
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__("MicroScalping", config)
        
        # Strategy-specific parameters
        self.timeframe = config.get('timeframe', '1s')
        self.profit_target = config.get('profit_target', 0.003)  # 0.3%
        self.stop_loss = config.get('stop_loss', 0.001)  # 0.1%
        self.max_hold_time = config.get('max_hold_time', 300)  # 5 minutes
        
        # Technical indicators
        self.ema_fast_period = 5
        self.ema_slow_period = 20
        self.rsi_period = 14
        
        # Price tracking
        self.tick_data = []
        self.last_prices = []
        
        logger.info("MicroScalping strategy initialized")
    
    async def generate_signal(self) -> Optional[TradeSignal]:
        """Generate micro scalping signals"""
        try:
            if not self.active or not self.current_market_data:
                return None
            
            # Check market conditions
            if not self.is_market_condition_suitable(self.current_market_data):
                return None
            
            # Calculate technical indicators
            indicators = await self._calculate_indicators()
            
            # Generate signal based on micro movements
            signal = await self._analyze_micro_movement(indicators)
            
            return signal
            
        except Exception as e:
            logger.error(f"Error generating micro scalping signal: {e}")
            return None
    
    async def update_market_data(self, data: Dict[str, Any]):
        """Update market data for micro scalping"""
        try:
            self.current_market_data = data
            
            # Track price movements
            if 'price' in data:
                self.last_prices.append(data['price'])
                
                # Keep only recent prices
                if len(self.last_prices) > 100:
                    self.last_prices = self.last_prices[-100:]
            
            # Track tick data
            if data.get('type') == 'trade':
                self.tick_data.append({
                    'price': data.get('price', 0.0),
                    'volume': data.get('quantity', 0.0),
                    'timestamp': data.get('timestamp', time.time() * 1000)
                })
                
                # Keep only recent ticks
                if len(self.tick_data) > 1000:
                    self.tick_data = self.tick_data[-1000:]
            
        except Exception as e:
            logger.error(f"Error updating market data: {e}")
    
    async def _calculate_indicators(self) -> Dict[str, float]:
        """Calculate technical indicators for micro scalping"""
        try:
            if len(self.last_prices) < self.ema_slow_period:
                return {}
            
            prices = np.array(self.last_prices)
            
            # EMA calculations
            ema_fast = self._calculate_ema(prices, self.ema_fast_period)
            ema_slow = self._calculate_ema(prices, self.ema_slow_period)
            
            # RSI calculation
            rsi = self._calculate_rsi(prices, self.rsi_period)
            
            # Price momentum
            momentum = (prices[-1] - prices[-5]) / prices[-5] if len(prices) >= 5 else 0.0
            
            # Volatility
            volatility = np.std(prices[-20:]) / np.mean(prices[-20:]) if len(prices) >= 20 else 0.0
            
            return {
                'ema_fast': ema_fast,
                'ema_slow': ema_slow,
                'rsi': rsi,
                'momentum': momentum,
                'volatility': volatility,
                'current_price': prices[-1]
            }
            
        except Exception as e:
            logger.error(f"Error calculating indicators: {e}")
            return {}
    
    async def _analyze_micro_movement(self, indicators: Dict[str, float]) -> Optional[TradeSignal]:
        """Analyze micro price movements for scalping opportunities"""
        try:
            if not indicators:
                return None
            
            current_price = indicators['current_price']
            ema_fast = indicators['ema_fast']
            ema_slow = indicators['ema_slow']
            rsi = indicators['rsi']
            momentum = indicators['momentum']
            
            signal_strength = 0.0
            side = None
            
            # EMA crossover signals
            if ema_fast > ema_slow:
                # Potential buy signal
                if momentum > 0.001:  # 0.1% positive momentum
                    signal_strength += 0.3
                    side = OrderSide.BUY
            elif ema_fast < ema_slow:
                # Potential sell signal
                if momentum < -0.001:  # 0.1% negative momentum
                    signal_strength += 0.3
                    side = OrderSide.SELL
            
            # RSI confirmation
            if side == OrderSide.BUY and rsi < 70:
                signal_strength += 0.2
            elif side == OrderSide.SELL and rsi > 30:
                signal_strength += 0.2
            
            # Volume confirmation
            if await self._check_volume_confirmation():
                signal_strength += 0.2
            
            # Tick analysis
            tick_signal = await self._analyze_tick_flow()
            if tick_signal:
                signal_strength += 0.3
                if not side:
                    side = tick_signal
            
            # Generate signal if strong enough
            if signal_strength >= self.min_signal_strength and side:
                # Calculate risk levels
                risk_levels = self.get_risk_levels(current_price)
                
                signal = TradeSignal(
                    symbol=self.current_market_data.get('symbol', 'BTCUSDT'),
                    side=side,
                    signal_strength=min(signal_strength, 1.0),
                    price_target=current_price,
                    stop_loss=risk_levels['stop_loss'],
                    take_profit=risk_levels['take_profit'],
                    urgency="HIGH",
                    strategy_id=self.name,
                    timestamp=time.time()
                )
                
                return signal
            
            return None
            
        except Exception as e:
            logger.error(f"Error analyzing micro movement: {e}")
            return None
    
    async def _check_volume_confirmation(self) -> bool:
        """Check if volume confirms the price movement"""
        try:
            if len(self.tick_data) < 10:
                return False
            
            recent_ticks = self.tick_data[-10:]
            volumes = [tick['volume'] for tick in recent_ticks]
            avg_volume = np.mean(volumes)
            recent_volume = volumes[-1]
            
            # Check if recent volume is above average
            return recent_volume > avg_volume * 1.2
            
        except Exception as e:
            logger.error(f"Error checking volume confirmation: {e}")
            return False
    
    async def _analyze_tick_flow(self) -> Optional[OrderSide]:
        """Analyze tick flow for buy/sell pressure"""
        try:
            if len(self.tick_data) < 20:
                return None
            
            recent_ticks = self.tick_data[-20:]
            
            # Analyze price direction in recent ticks
            buy_pressure = 0
            sell_pressure = 0
            
            for i in range(1, len(recent_ticks)):
                if recent_ticks[i]['price'] > recent_ticks[i-1]['price']:
                    buy_pressure += recent_ticks[i]['volume']
                elif recent_ticks[i]['price'] < recent_ticks[i-1]['price']:
                    sell_pressure += recent_ticks[i]['volume']
            
            total_pressure = buy_pressure + sell_pressure
            if total_pressure == 0:
                return None
            
            buy_ratio = buy_pressure / total_pressure
            
            # Strong buy pressure
            if buy_ratio > 0.65:
                return OrderSide.BUY
            # Strong sell pressure
            elif buy_ratio < 0.35:
                return OrderSide.SELL
            
            return None
            
        except Exception as e:
            logger.error(f"Error analyzing tick flow: {e}")
            return None
    
    def _calculate_ema(self, prices: np.ndarray, period: int) -> float:
        """Calculate Exponential Moving Average"""
        try:
            alpha = 2 / (period + 1)
            ema = prices[0]
            
            for price in prices[1:]:
                ema = alpha * price + (1 - alpha) * ema
            
            return ema
            
        except Exception as e:
            logger.error(f"Error calculating EMA: {e}")
            return prices[-1] if len(prices) > 0 else 0.0
    
    def _calculate_rsi(self, prices: np.ndarray, period: int = 14) -> float:
        """Calculate RSI"""
        try:
            if len(prices) < period + 1:
                return 50.0
            
            deltas = np.diff(prices)
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


class OrderBookImbalanceStrategy(BaseStrategy):
    """
    Strategy based on order book imbalance analysis
    Detects buying/selling pressure from order book data
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__("OrderBookImbalance", config)
        
        # Strategy parameters
        self.imbalance_threshold = config.get('imbalance_threshold', 0.7)
        self.depth_levels = config.get('depth_levels', 10)
        self.min_order_size = config.get('min_order_size', 1000)
        
        # Order book tracking
        self.order_book_history = []
        self.imbalance_history = []
        
        logger.info("OrderBookImbalance strategy initialized")
    
    async def generate_signal(self) -> Optional[TradeSignal]:
        """Generate signals based on order book imbalance"""
        try:
            if not self.active or not self.current_market_data:
                return None
            
            # Only process order book data
            if self.current_market_data.get('type') != 'orderbook':
                return None
            
            # Calculate order book imbalance
            imbalance_data = await self._calculate_order_book_imbalance()
            
            if not imbalance_data:
                return None
            
            # Generate signal based on imbalance
            signal = await self._generate_imbalance_signal(imbalance_data)
            
            return signal
            
        except Exception as e:
            logger.error(f"Error generating order book imbalance signal: {e}")
            return None
    
    async def update_market_data(self, data: Dict[str, Any]):
        """Update market data for order book analysis"""
        try:
            self.current_market_data = data
            
            # Store order book data
            if data.get('type') == 'orderbook':
                self.order_book_history.append({
                    'bids': data.get('bids', []),
                    'asks': data.get('asks', []),
                    'timestamp': data.get('timestamp', time.time() * 1000)
                })
                
                # Keep limited history
                if len(self.order_book_history) > 100:
                    self.order_book_history = self.order_book_history[-100:]
            
        except Exception as e:
            logger.error(f"Error updating market data: {e}")
    
    async def _calculate_order_book_imbalance(self) -> Optional[Dict[str, float]]:
        """Calculate order book imbalance metrics"""
        try:
            bids = self.current_market_data.get('bids', [])
            asks = self.current_market_data.get('asks', [])
            
            if not bids or not asks:
                return None
            
            # Calculate bid/ask volumes at different levels
            bid_volumes = []
            ask_volumes = []
            
            for i in range(min(self.depth_levels, len(bids), len(asks))):
                bid_volumes.append(bids[i][1])
                ask_volumes.append(asks[i][1])
            
            total_bid_volume = sum(bid_volumes)
            total_ask_volume = sum(ask_volumes)
            total_volume = total_bid_volume + total_ask_volume
            
            if total_volume == 0:
                return None
            
            # Calculate imbalance ratio
            imbalance = (total_bid_volume - total_ask_volume) / total_volume
            
            # Calculate weighted imbalance (closer levels have more weight)
            weighted_bid = sum(vol / (i + 1) for i, vol in enumerate(bid_volumes))
            weighted_ask = sum(vol / (i + 1) for i, vol in enumerate(ask_volumes))
            weighted_total = weighted_bid + weighted_ask
            
            weighted_imbalance = (weighted_bid - weighted_ask) / weighted_total if weighted_total > 0 else 0
            
            # Calculate spread metrics
            best_bid = bids[0][0]
            best_ask = asks[0][0]
            spread = best_ask - best_bid
            spread_pct = spread / best_bid
            
            imbalance_data = {
                'imbalance': imbalance,
                'weighted_imbalance': weighted_imbalance,
                'total_bid_volume': total_bid_volume,
                'total_ask_volume': total_ask_volume,
                'spread_pct': spread_pct,
                'best_bid': best_bid,
                'best_ask': best_ask,
                'mid_price': (best_bid + best_ask) / 2
            }
            
            # Store in history
            self.imbalance_history.append(imbalance_data)
            if len(self.imbalance_history) > 50:
                self.imbalance_history = self.imbalance_history[-50:]
            
            return imbalance_data
            
        except Exception as e:
            logger.error(f"Error calculating order book imbalance: {e}")
            return None
    
    async def _generate_imbalance_signal(self, imbalance_data: Dict[str, float]) -> Optional[TradeSignal]:
        """Generate signal based on order book imbalance"""
        try:
            imbalance = imbalance_data['imbalance']
            weighted_imbalance = imbalance_data['weighted_imbalance']
            total_bid_volume = imbalance_data['total_bid_volume']
            total_ask_volume = imbalance_data['total_ask_volume']
            mid_price = imbalance_data['mid_price']
            
            signal_strength = 0.0
            side = None
            
            # Strong bid imbalance (buying pressure)
            if imbalance > self.imbalance_threshold:
                signal_strength += 0.4
                side = OrderSide.BUY
                
                # Additional confirmation from weighted imbalance
                if weighted_imbalance > self.imbalance_threshold * 0.8:
                    signal_strength += 0.2
            
            # Strong ask imbalance (selling pressure)
            elif imbalance < -self.imbalance_threshold:
                signal_strength += 0.4
                side = OrderSide.SELL
                
                # Additional confirmation from weighted imbalance
                if weighted_imbalance < -self.imbalance_threshold * 0.8:
                    signal_strength += 0.2
            
            # Check volume requirements
            min_volume = self.min_order_size
            if (total_bid_volume + total_ask_volume) < min_volume:
                return None
            
            # Historical imbalance trend
            if len(self.imbalance_history) >= 3:
                recent_imbalances = [data['imbalance'] for data in self.imbalance_history[-3:]]
                
                # Consistent imbalance direction
                if side == OrderSide.BUY and all(im > 0.3 for im in recent_imbalances):
                    signal_strength += 0.2
                elif side == OrderSide.SELL and all(im < -0.3 for im in recent_imbalances):
                    signal_strength += 0.2
            
            # Spread check
            if imbalance_data['spread_pct'] < 0.002:  # Tight spread
                signal_strength += 0.2
            
            # Generate signal if strong enough
            if signal_strength >= self.min_signal_strength and side:
                # Calculate risk levels
                stop_loss_pct = 0.002  # 0.2% for tight scalping
                take_profit_pct = 0.004  # 0.4% target
                
                if side == OrderSide.BUY:
                    stop_loss = mid_price * (1 - stop_loss_pct)
                    take_profit = mid_price * (1 + take_profit_pct)
                else:
                    stop_loss = mid_price * (1 + stop_loss_pct)
                    take_profit = mid_price * (1 - take_profit_pct)
                
                signal = TradeSignal(
                    symbol=self.current_market_data.get('symbol', 'BTCUSDT'),
                    side=side,
                    signal_strength=min(signal_strength, 1.0),
                    price_target=mid_price,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    urgency="URGENT",  # Order book signals are time-sensitive
                    strategy_id=self.name,
                    timestamp=time.time()
                )
                
                return signal
            
            return None
            
        except Exception as e:
            logger.error(f"Error generating imbalance signal: {e}")
            return None 