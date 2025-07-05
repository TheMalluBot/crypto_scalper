"""
Risk Manager for crypto scalping
Implements comprehensive risk controls and monitoring
"""

import time
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum

import numpy as np
from loguru import logger

from .order_manager import TradeSignal, OrderSide
from .position_manager import Position


class RiskLevel(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class RiskMetrics:
    """Risk metrics snapshot"""
    portfolio_var: float = 0.0  # Value at Risk
    max_drawdown: float = 0.0
    daily_pnl: float = 0.0
    volatility: float = 0.0
    beta: float = 0.0
    sharpe_ratio: float = 0.0
    correlation_risk: float = 0.0
    concentration_risk: float = 0.0
    liquidity_risk: float = 0.0
    timestamp: float = 0.0


@dataclass
class RiskAlert:
    """Risk alert"""
    level: RiskLevel
    message: str
    metric: str
    value: float
    threshold: float
    timestamp: float


class RiskManager:
    """
    Advanced risk management system
    Implements multiple layers of risk controls
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Risk limits from config
        self.max_position_size = config.get('max_position_size', 0.02)
        self.max_daily_loss = config.get('max_daily_loss', 0.05)
        self.max_drawdown = config.get('max_drawdown', 0.10)
        self.stop_loss_pct = config.get('stop_loss_pct', 0.005)
        self.take_profit_pct = config.get('take_profit_pct', 0.015)
        self.risk_reward_ratio = config.get('risk_reward_ratio', 3.0)
        
        # Risk tracking
        self.daily_pnl = 0.0
        self.session_start_time = time.time()
        self.current_drawdown = 0.0
        self.peak_portfolio_value = 0.0
        
        # Risk metrics history
        self.risk_metrics_history: List[RiskMetrics] = []
        self.risk_alerts: List[RiskAlert] = []
        
        # Emergency controls
        self.emergency_stop_triggered = False
        self.trading_halted = False
        
        # Market volatility tracking
        self.market_volatility = {}
        self.correlation_matrix = {}
        
        logger.info("Risk Manager initialized")
    
    async def check_trade_risk(self, signal: TradeSignal) -> bool:
        """
        Check if a trade signal passes risk controls
        
        Args:
            signal: Trading signal to validate
            
        Returns:
            True if trade is allowed, False otherwise
        """
        try:
            # Check if trading is halted
            if self.trading_halted or self.emergency_stop_triggered:
                logger.warning("Trading halted - rejecting signal")
                return False
            
            # Check position size limits
            if not await self._check_position_size_limits(signal):
                return False
            
            # Check daily loss limits
            if not await self._check_daily_loss_limits():
                return False
            
            # Check drawdown limits
            if not await self._check_drawdown_limits():
                return False
            
            # Check risk-reward ratio
            if not await self._check_risk_reward_ratio(signal):
                return False
            
            # Check market conditions
            if not await self._check_market_conditions(signal):
                return False
            
            # Check correlation risk
            if not await self._check_correlation_risk(signal):
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error in risk check: {e}")
            return False
    
    async def check_risk_limits(self) -> Dict[str, Any]:
        """
        Check all risk limits and return status
        
        Returns:
            Risk status dictionary
        """
        try:
            risk_status = {
                'emergency_stop': False,
                'trading_halt': False,
                'risk_level': RiskLevel.LOW,
                'alerts': []
            }
            
            # Check daily loss
            if abs(self.daily_pnl) > self.max_daily_loss:
                risk_status['emergency_stop'] = True
                risk_status['risk_level'] = RiskLevel.CRITICAL
                
                alert = RiskAlert(
                    level=RiskLevel.CRITICAL,
                    message=f"Daily loss limit exceeded: {self.daily_pnl:.2%}",
                    metric="daily_pnl",
                    value=abs(self.daily_pnl),
                    threshold=self.max_daily_loss,
                    timestamp=time.time()
                )
                risk_status['alerts'].append(alert)
                self.risk_alerts.append(alert)
            
            # Check drawdown
            if self.current_drawdown > self.max_drawdown:
                risk_status['emergency_stop'] = True
                risk_status['risk_level'] = RiskLevel.CRITICAL
                
                alert = RiskAlert(
                    level=RiskLevel.CRITICAL,
                    message=f"Maximum drawdown exceeded: {self.current_drawdown:.2%}",
                    metric="drawdown",
                    value=self.current_drawdown,
                    threshold=self.max_drawdown,
                    timestamp=time.time()
                )
                risk_status['alerts'].append(alert)
                self.risk_alerts.append(alert)
            
            # Check volatility
            volatility_risk = await self._check_volatility_risk()
            if volatility_risk['level'] != RiskLevel.LOW:
                risk_status['risk_level'] = max(risk_status['risk_level'], volatility_risk['level'])
                risk_status['alerts'].extend(volatility_risk['alerts'])
            
            # Update emergency stop status
            if risk_status['emergency_stop']:
                self.emergency_stop_triggered = True
            
            return risk_status
            
        except Exception as e:
            logger.error(f"Error checking risk limits: {e}")
            return {'emergency_stop': False, 'trading_halt': False, 'risk_level': RiskLevel.LOW, 'alerts': []}
    
    async def update_daily_pnl(self, pnl_change: float):
        """Update daily PnL tracking"""
        self.daily_pnl += pnl_change
        
        # Check if we need to reset for new day
        current_time = time.time()
        if current_time - self.session_start_time > 86400:  # 24 hours
            self.daily_pnl = pnl_change
            self.session_start_time = current_time
    
    async def update_portfolio_value(self, portfolio_value: float):
        """Update portfolio value and calculate drawdown"""
        if portfolio_value > self.peak_portfolio_value:
            self.peak_portfolio_value = portfolio_value
            self.current_drawdown = 0.0
        else:
            self.current_drawdown = (self.peak_portfolio_value - portfolio_value) / self.peak_portfolio_value
    
    async def calculate_var(self, positions: List[Position], confidence: float = 0.95) -> float:
        """
        Calculate Value at Risk for the portfolio
        
        Args:
            positions: Current positions
            confidence: VaR confidence level
            
        Returns:
            VaR value
        """
        try:
            if not positions:
                return 0.0
            
            # Get position values
            position_values = []
            for position in positions:
                value = position.size * position.current_price
                if position.side == OrderSide.SELL:
                    value = -value
                position_values.append(value)
            
            # Calculate portfolio volatility (simplified)
            # In practice, this would use historical returns and correlation matrix
            portfolio_value = sum(abs(val) for val in position_values)
            volatility = 0.02  # 2% daily volatility assumption
            
            # VaR calculation
            z_score = 1.645 if confidence == 0.95 else 2.326  # 95% or 99%
            var = portfolio_value * volatility * z_score
            
            return var
            
        except Exception as e:
            logger.error(f"Error calculating VaR: {e}")
            return 0.0
    
    async def _check_position_size_limits(self, signal: TradeSignal) -> bool:
        """Check position size limits"""
        # This would check against current portfolio allocation
        # For now, return True as position sizing is handled by PositionManager
        return True
    
    async def _check_daily_loss_limits(self) -> bool:
        """Check daily loss limits"""
        if abs(self.daily_pnl) >= self.max_daily_loss * 0.8:  # 80% of limit
            logger.warning(f"Approaching daily loss limit: {self.daily_pnl:.2%}")
            
            if abs(self.daily_pnl) >= self.max_daily_loss:
                logger.critical("Daily loss limit exceeded!")
                return False
        
        return True
    
    async def _check_drawdown_limits(self) -> bool:
        """Check drawdown limits"""
        if self.current_drawdown >= self.max_drawdown * 0.8:  # 80% of limit
            logger.warning(f"Approaching drawdown limit: {self.current_drawdown:.2%}")
            
            if self.current_drawdown >= self.max_drawdown:
                logger.critical("Maximum drawdown exceeded!")
                return False
        
        return True
    
    async def _check_risk_reward_ratio(self, signal: TradeSignal) -> bool:
        """Check risk-reward ratio"""
        if signal.stop_loss and signal.take_profit:
            # Calculate risk and reward
            if signal.price_target:
                entry_price = signal.price_target
            else:
                entry_price = 50000.0  # Placeholder - would get current price
            
            risk = abs(entry_price - signal.stop_loss)
            reward = abs(signal.take_profit - entry_price)
            
            if risk > 0:
                rr_ratio = reward / risk
                if rr_ratio < self.risk_reward_ratio:
                    logger.warning(f"Poor risk-reward ratio: {rr_ratio:.2f} < {self.risk_reward_ratio}")
                    return False
        
        return True
    
    async def _check_market_conditions(self, signal: TradeSignal) -> bool:
        """Check market conditions"""
        # Check volatility
        symbol_volatility = self.market_volatility.get(signal.symbol, 0.02)
        
        # Reject trades in extremely volatile conditions
        if symbol_volatility > 0.10:  # 10% volatility
            logger.warning(f"High volatility for {signal.symbol}: {symbol_volatility:.2%}")
            return False
        
        return True
    
    async def _check_correlation_risk(self, signal: TradeSignal) -> bool:
        """Check correlation risk with existing positions"""
        # This would check correlation with existing positions
        # For now, return True
        return True
    
    async def _check_volatility_risk(self) -> Dict[str, Any]:
        """Check volatility-based risk"""
        try:
            # Calculate average market volatility
            if not self.market_volatility:
                return {'level': RiskLevel.LOW, 'alerts': []}
            
            avg_volatility = np.mean(list(self.market_volatility.values()))
            
            alerts = []
            risk_level = RiskLevel.LOW
            
            if avg_volatility > 0.05:  # 5% threshold
                risk_level = RiskLevel.MEDIUM
                
                alert = RiskAlert(
                    level=RiskLevel.MEDIUM,
                    message=f"Elevated market volatility: {avg_volatility:.2%}",
                    metric="volatility",
                    value=avg_volatility,
                    threshold=0.05,
                    timestamp=time.time()
                )
                alerts.append(alert)
                self.risk_alerts.append(alert)
            
            if avg_volatility > 0.10:  # 10% threshold
                risk_level = RiskLevel.HIGH
                
                alert = RiskAlert(
                    level=RiskLevel.HIGH,
                    message=f"High market volatility: {avg_volatility:.2%}",
                    metric="volatility",
                    value=avg_volatility,
                    threshold=0.10,
                    timestamp=time.time()
                )
                alerts.append(alert)
                self.risk_alerts.append(alert)
            
            return {'level': risk_level, 'alerts': alerts}
            
        except Exception as e:
            logger.error(f"Error checking volatility risk: {e}")
            return {'level': RiskLevel.LOW, 'alerts': []}
    
    def update_market_volatility(self, symbol: str, volatility: float):
        """Update market volatility for a symbol"""
        self.market_volatility[symbol] = volatility
    
    def halt_trading(self, reason: str):
        """Halt all trading"""
        self.trading_halted = True
        logger.critical(f"Trading halted: {reason}")
    
    def resume_trading(self):
        """Resume trading"""
        self.trading_halted = False
        self.emergency_stop_triggered = False
        logger.info("Trading resumed")
    
    def get_risk_metrics(self) -> RiskMetrics:
        """Get current risk metrics"""
        try:
            volatility = np.mean(list(self.market_volatility.values())) if self.market_volatility else 0.0
            
            return RiskMetrics(
                daily_pnl=self.daily_pnl,
                max_drawdown=self.current_drawdown,
                volatility=volatility,
                concentration_risk=self._calculate_concentration_risk(),
                timestamp=time.time()
            )
            
        except Exception as e:
            logger.error(f"Error getting risk metrics: {e}")
            return RiskMetrics(timestamp=time.time())
    
    def _calculate_concentration_risk(self) -> float:
        """Calculate portfolio concentration risk"""
        # Simplified concentration risk calculation
        # In practice, this would analyze position distribution
        return 0.0
    
    def get_recent_alerts(self, hours: int = 24) -> List[RiskAlert]:
        """Get recent risk alerts"""
        cutoff_time = time.time() - (hours * 3600)
        return [alert for alert in self.risk_alerts if alert.timestamp > cutoff_time]
    
    def get_risk_summary(self) -> Dict[str, Any]:
        """Get risk management summary"""
        return {
            'emergency_stop_triggered': self.emergency_stop_triggered,
            'trading_halted': self.trading_halted,
            'daily_pnl': self.daily_pnl,
            'current_drawdown': self.current_drawdown,
            'max_daily_loss_limit': self.max_daily_loss,
            'max_drawdown_limit': self.max_drawdown,
            'recent_alerts': len(self.get_recent_alerts()),
            'avg_market_volatility': np.mean(list(self.market_volatility.values())) if self.market_volatility else 0.0
        } 