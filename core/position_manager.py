"""
Position Manager for crypto scalping
Handles position tracking, sizing, and portfolio management
"""

import time
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum

import numpy as np
from loguru import logger

from .order_manager import Order, OrderSide, TradeSignal


@dataclass
class Position:
    """Position representation"""
    symbol: str
    side: OrderSide
    size: float
    entry_price: float
    current_price: float
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    entry_time: float = 0.0
    update_time: float = 0.0


class PositionManager:
    """
    Position manager for portfolio tracking and position sizing
    Implements Kelly Criterion and risk-based position sizing
    """
    
    def __init__(self):
        # Position tracking
        self.positions: Dict[str, Position] = {}
        self.closed_positions: List[Position] = []
        
        # Portfolio metrics
        self.portfolio_value = 100000.0  # Starting value in USDT
        self.available_balance = 100000.0
        self.total_unrealized_pnl = 0.0
        self.total_realized_pnl = 0.0
        
        # Risk parameters
        self.max_position_size = 0.02  # 2% per position
        self.max_portfolio_risk = 0.10  # 10% max risk
        self.kelly_multiplier = 0.25  # Conservative Kelly
        
        # Performance tracking
        self.trade_history = []
        self.win_rate = 0.0
        self.avg_win = 0.0
        self.avg_loss = 0.0
        
        logger.info("Position Manager initialized")
    
    async def calculate_position_size(self, signal: TradeSignal) -> float:
        """
        Calculate optimal position size using Kelly Criterion and risk management
        
        Args:
            signal: Trading signal with risk/reward parameters
            
        Returns:
            Position size in base currency
        """
        try:
            # Get current price for the symbol
            current_price = await self._get_current_price(signal.symbol)
            
            # Calculate base position size (% of portfolio)
            base_size_pct = min(self.max_position_size, self._calculate_kelly_size(signal))
            
            # Convert to actual size
            position_value = self.available_balance * base_size_pct
            position_size = position_value / current_price
            
            # Apply risk adjustments
            adjusted_size = self._apply_risk_adjustments(position_size, signal)
            
            logger.debug(f"Position size calculated: {adjusted_size:.6f} {signal.symbol}")
            return adjusted_size
            
        except Exception as e:
            logger.error(f"Error calculating position size: {e}")
            return 0.0
    
    async def update_position(self, order: Order):
        """Update position based on filled order"""
        try:
            symbol = order.symbol
            existing_position = self.positions.get(symbol)
            
            if not existing_position:
                # New position
                position = Position(
                    symbol=symbol,
                    side=order.side,
                    size=order.filled_quantity,
                    entry_price=order.avg_price,
                    current_price=order.avg_price,
                    entry_time=order.timestamp,
                    update_time=time.time()
                )
                self.positions[symbol] = position
                
                # Update available balance
                position_value = order.filled_quantity * order.avg_price
                if order.side == OrderSide.BUY:
                    self.available_balance -= position_value
                else:
                    self.available_balance += position_value
                
                logger.info(f"New position opened: {symbol} {order.side.value} {order.filled_quantity}")
                
            else:
                # Update existing position
                await self._update_existing_position(existing_position, order)
            
            # Update portfolio metrics
            await self._update_portfolio_metrics()
            
        except Exception as e:
            logger.error(f"Error updating position: {e}")
    
    async def update_from_account_data(self, account_data: Dict[str, Any]):
        """Update positions from account stream data"""
        try:
            # Update balance
            if 'B' in account_data:  # Balance update
                for balance in account_data['B']:
                    if balance['a'] == 'USDT':
                        self.available_balance = float(balance['f'])  # free balance
            
            # Update position prices
            await self._update_position_prices()
            
        except Exception as e:
            logger.error(f"Error updating from account data: {e}")
    
    async def close_position(self, symbol: str) -> bool:
        """Close a specific position"""
        try:
            position = self.positions.get(symbol)
            if not position:
                logger.warning(f"No position found for {symbol}")
                return False
            
            # Calculate realized PnL
            current_price = await self._get_current_price(symbol)
            realized_pnl = self._calculate_realized_pnl(position, current_price)
            
            # Update metrics
            self.total_realized_pnl += realized_pnl
            position.realized_pnl = realized_pnl
            
            # Move to closed positions
            self.closed_positions.append(position)
            del self.positions[symbol]
            
            # Update trade history
            self.trade_history.append({
                'symbol': symbol,
                'side': position.side.value,
                'size': position.size,
                'entry_price': position.entry_price,
                'exit_price': current_price,
                'pnl': realized_pnl,
                'duration': time.time() - position.entry_time,
                'timestamp': time.time()
            })
            
            # Update performance metrics
            await self._update_performance_metrics()
            
            logger.info(f"Position closed: {symbol}, PnL: {realized_pnl:.2f}")
            return True
            
        except Exception as e:
            logger.error(f"Error closing position {symbol}: {e}")
            return False
    
    async def close_all_positions(self) -> int:
        """Close all open positions"""
        closed_count = 0
        symbols = list(self.positions.keys())
        
        for symbol in symbols:
            if await self.close_position(symbol):
                closed_count += 1
        
        logger.info(f"Closed {closed_count} positions")
        return closed_count
    
    def _calculate_kelly_size(self, signal: TradeSignal) -> float:
        """Calculate Kelly Criterion position size"""
        try:
            # Need win rate and average win/loss
            if self.win_rate == 0 or self.avg_win == 0 or self.avg_loss == 0:
                return self.max_position_size * 0.5  # Conservative default
            
            # Kelly formula: f = (bp - q) / b
            # where b = odds (avg_win/avg_loss), p = win_rate, q = loss_rate
            b = abs(self.avg_win / self.avg_loss)
            p = self.win_rate
            q = 1 - p
            
            kelly_fraction = (b * p - q) / b
            
            # Apply conservative multiplier and cap
            kelly_size = max(0, min(kelly_fraction * self.kelly_multiplier, self.max_position_size))
            
            return kelly_size
            
        except Exception as e:
            logger.error(f"Error calculating Kelly size: {e}")
            return self.max_position_size * 0.5
    
    def _apply_risk_adjustments(self, position_size: float, signal: TradeSignal) -> float:
        """Apply additional risk adjustments to position size"""
        try:
            # Adjust based on signal strength
            strength_multiplier = min(signal.signal_strength, 1.0)
            adjusted_size = position_size * strength_multiplier
            
            # Check portfolio concentration
            current_exposure = sum(
                abs(pos.size * pos.current_price) 
                for pos in self.positions.values()
            ) / self.portfolio_value
            
            if current_exposure > self.max_portfolio_risk:
                exposure_reduction = (self.max_portfolio_risk / current_exposure) * 0.5
                adjusted_size *= exposure_reduction
            
            # Minimum size filter
            min_size = 0.001  # Minimum trade size
            if adjusted_size < min_size:
                return 0.0
            
            return adjusted_size
            
        except Exception as e:
            logger.error(f"Error applying risk adjustments: {e}")
            return position_size * 0.5
    
    async def _update_existing_position(self, position: Position, order: Order):
        """Update existing position with new order"""
        try:
            if order.side == position.side:
                # Adding to position
                total_cost = (position.size * position.entry_price) + (order.filled_quantity * order.avg_price)
                total_size = position.size + order.filled_quantity
                position.entry_price = total_cost / total_size
                position.size = total_size
                
            else:
                # Reducing position
                if order.filled_quantity >= position.size:
                    # Position closed or reversed
                    realized_pnl = self._calculate_realized_pnl(position, order.avg_price)
                    self.total_realized_pnl += realized_pnl
                    
                    if order.filled_quantity > position.size:
                        # Position reversed
                        remaining_size = order.filled_quantity - position.size
                        position.side = order.side
                        position.size = remaining_size
                        position.entry_price = order.avg_price
                    else:
                        # Position closed
                        self.closed_positions.append(position)
                        del self.positions[position.symbol]
                        return
                else:
                    # Partial close
                    close_ratio = order.filled_quantity / position.size
                    realized_pnl = self._calculate_realized_pnl(position, order.avg_price) * close_ratio
                    self.total_realized_pnl += realized_pnl
                    position.size -= order.filled_quantity
            
            position.update_time = time.time()
            
        except Exception as e:
            logger.error(f"Error updating existing position: {e}")
    
    def _calculate_realized_pnl(self, position: Position, exit_price: float) -> float:
        """Calculate realized PnL for a position"""
        if position.side == OrderSide.BUY:
            return (exit_price - position.entry_price) * position.size
        else:
            return (position.entry_price - exit_price) * position.size
    
    async def _update_position_prices(self):
        """Update current prices for all positions"""
        for position in self.positions.values():
            try:
                current_price = await self._get_current_price(position.symbol)
                position.current_price = current_price
                
                # Calculate unrealized PnL
                if position.side == OrderSide.BUY:
                    position.unrealized_pnl = (current_price - position.entry_price) * position.size
                else:
                    position.unrealized_pnl = (position.entry_price - current_price) * position.size
                
            except Exception as e:
                logger.error(f"Error updating price for {position.symbol}: {e}")
    
    async def _update_portfolio_metrics(self):
        """Update portfolio-level metrics"""
        try:
            # Calculate total unrealized PnL
            self.total_unrealized_pnl = sum(pos.unrealized_pnl for pos in self.positions.values())
            
            # Update portfolio value
            position_value = sum(
                abs(pos.size * pos.current_price) 
                for pos in self.positions.values()
            )
            self.portfolio_value = self.available_balance + position_value + self.total_unrealized_pnl
            
        except Exception as e:
            logger.error(f"Error updating portfolio metrics: {e}")
    
    async def _update_performance_metrics(self):
        """Update performance metrics from trade history"""
        try:
            if not self.trade_history:
                return
            
            # Calculate win rate
            wins = [trade for trade in self.trade_history if trade['pnl'] > 0]
            losses = [trade for trade in self.trade_history if trade['pnl'] < 0]
            
            self.win_rate = len(wins) / len(self.trade_history)
            self.avg_win = np.mean([trade['pnl'] for trade in wins]) if wins else 0
            self.avg_loss = np.mean([trade['pnl'] for trade in losses]) if losses else 0
            
        except Exception as e:
            logger.error(f"Error updating performance metrics: {e}")
    
    async def _get_current_price(self, symbol: str) -> float:
        """Get current market price for symbol"""
        # This would normally fetch from the data processor or API
        # For now, return a placeholder
        return 50000.0  # Placeholder price
    
    def get_positions(self) -> List[Position]:
        """Get all current positions"""
        return list(self.positions.values())
    
    def get_position(self, symbol: str) -> Optional[Position]:
        """Get position for specific symbol"""
        return self.positions.get(symbol)
    
    def get_portfolio_summary(self) -> Dict[str, Any]:
        """Get portfolio summary"""
        return {
            'total_value': self.portfolio_value,
            'available_balance': self.available_balance,
            'total_unrealized_pnl': self.total_unrealized_pnl,
            'total_realized_pnl': self.total_realized_pnl,
            'open_positions': len(self.positions),
            'win_rate': self.win_rate,
            'total_trades': len(self.trade_history)
        }
    
    def get_risk_metrics(self) -> Dict[str, float]:
        """Get risk metrics"""
        try:
            if not self.trade_history:
                return {}
            
            pnl_series = np.array([trade['pnl'] for trade in self.trade_history])
            
            return {
                'sharpe_ratio': np.mean(pnl_series) / np.std(pnl_series) if np.std(pnl_series) > 0 else 0,
                'max_drawdown': self._calculate_max_drawdown(),
                'profit_factor': abs(self.avg_win / self.avg_loss) if self.avg_loss != 0 else 0,
                'win_rate': self.win_rate,
                'avg_win': self.avg_win,
                'avg_loss': self.avg_loss
            }
            
        except Exception as e:
            logger.error(f"Error calculating risk metrics: {e}")
            return {}
    
    def _calculate_max_drawdown(self) -> float:
        """Calculate maximum drawdown"""
        try:
            if not self.trade_history:
                return 0.0
            
            cumulative_pnl = np.cumsum([trade['pnl'] for trade in self.trade_history])
            peak = np.maximum.accumulate(cumulative_pnl)
            drawdown = (cumulative_pnl - peak) / peak
            
            return abs(np.min(drawdown))
            
        except Exception as e:
            logger.error(f"Error calculating max drawdown: {e}")
            return 0.0 