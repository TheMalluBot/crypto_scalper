"""
Order Manager for high-frequency crypto trading
Handles order execution, management, and tracking
"""

import asyncio
import time
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum
import uuid

from loguru import logger
import numpy as np

from ..data.binance_client import BinanceClient


class OrderType(Enum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP_LOSS = "STOP_LOSS_LIMIT"
    TAKE_PROFIT = "TAKE_PROFIT_LIMIT"
    OCO = "OCO"


class OrderSide(Enum):
    BUY = "BUY"
    SELL = "SELL"


class OrderStatus(Enum):
    NEW = "NEW"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    FILLED = "FILLED"
    CANCELED = "CANCELED"
    REJECTED = "REJECTED"
    EXPIRED = "EXPIRED"


@dataclass
class Order:
    """Order representation"""
    id: str
    client_order_id: str
    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: float
    price: Optional[float]
    stop_price: Optional[float]
    status: OrderStatus
    filled_quantity: float = 0.0
    avg_price: float = 0.0
    commission: float = 0.0
    timestamp: float = 0.0
    update_time: float = 0.0


@dataclass
class TradeSignal:
    """Trading signal from strategies"""
    symbol: str
    side: OrderSide
    signal_strength: float
    price_target: Optional[float]
    stop_loss: Optional[float]
    take_profit: Optional[float]
    urgency: str = "NORMAL"  # LOW, NORMAL, HIGH, URGENT
    strategy_id: str = ""
    timestamp: float = 0.0


class OrderManager:
    """
    High-performance order manager
    Designed for low-latency execution
    """
    
    def __init__(self, binance_client: BinanceClient):
        self.client = binance_client
        
        # Order tracking
        self.active_orders: Dict[str, Order] = {}
        self.order_history: List[Order] = []
        
        # Execution metrics
        self.execution_latency = []
        self.slippage_tracking = []
        
        # Rate limiting
        self.order_timestamps = []
        self.max_orders_per_second = 10
        
        logger.info("Order Manager initialized")
    
    async def place_order(self, signal: TradeSignal, position_size: float) -> Optional[Order]:
        """
        Place order based on trading signal
        
        Args:
            signal: Trading signal
            position_size: Position size to trade
            
        Returns:
            Order object if successful, None otherwise
        """
        try:
            # Rate limiting check
            if not await self._check_rate_limits():
                logger.warning("Rate limit exceeded, skipping order")
                return None
            
            # Determine order type based on signal urgency
            order_type = self._determine_order_type(signal)
            
            # Calculate order parameters
            price = await self._calculate_order_price(signal, order_type)
            quantity = self._calculate_quantity(signal.symbol, position_size)
            
            # Create order
            order = await self._create_order(
                symbol=signal.symbol,
                side=signal.side,
                order_type=order_type,
                quantity=quantity,
                price=price,
                signal=signal
            )
            
            if order:
                # Track order
                self.active_orders[order.client_order_id] = order
                
                # Set up stop loss and take profit if specified
                if signal.stop_loss or signal.take_profit:
                    await self._setup_protection_orders(order, signal)
                
                logger.info(f"Order placed: {order.symbol} {order.side.value} "
                           f"{order.quantity} @ {order.price}")
            
            return order
            
        except Exception as e:
            logger.error(f"Error placing order: {e}")
            return None
    
    async def cancel_order(self, order_id: str) -> bool:
        """Cancel an active order"""
        try:
            order = self.active_orders.get(order_id)
            if not order:
                logger.warning(f"Order not found: {order_id}")
                return False
            
            # Cancel on exchange
            result = await self.client.cancel_order(
                symbol=order.symbol,
                origClientOrderId=order.client_order_id
            )
            
            if result:
                order.status = OrderStatus.CANCELED
                order.update_time = time.time()
                
                # Move to history
                self.order_history.append(order)
                del self.active_orders[order_id]
                
                logger.info(f"Order canceled: {order_id}")
                return True
            
        except Exception as e:
            logger.error(f"Error canceling order {order_id}: {e}")
        
        return False
    
    async def cancel_all_orders(self) -> int:
        """Cancel all active orders"""
        canceled_count = 0
        
        # Get all active order IDs
        order_ids = list(self.active_orders.keys())
        
        # Cancel in parallel
        cancel_tasks = [self.cancel_order(order_id) for order_id in order_ids]
        results = await asyncio.gather(*cancel_tasks, return_exceptions=True)
        
        for result in results:
            if result is True:
                canceled_count += 1
            elif isinstance(result, Exception):
                logger.error(f"Error in bulk cancel: {result}")
        
        logger.info(f"Canceled {canceled_count} orders")
        return canceled_count
    
    async def update_order_status(self, order_update: Dict[str, Any]):
        """Update order status from exchange stream"""
        try:
            client_order_id = order_update.get('c')  # clientOrderId
            order = self.active_orders.get(client_order_id)
            
            if not order:
                return
            
            # Update order fields
            order.status = OrderStatus(order_update.get('X'))  # orderStatus
            order.filled_quantity = float(order_update.get('z', 0))  # cumulative filled qty
            order.avg_price = float(order_update.get('Z', 0)) / order.filled_quantity if order.filled_quantity > 0 else 0
            order.update_time = order_update.get('T', time.time() * 1000) / 1000  # transaction time
            
            # Calculate execution metrics
            if order.status == OrderStatus.FILLED:
                await self._record_execution_metrics(order)
                
                # Move to history
                self.order_history.append(order)
                del self.active_orders[client_order_id]
                
                logger.info(f"Order filled: {order.symbol} {order.side.value} "
                           f"{order.filled_quantity} @ {order.avg_price}")
            
        except Exception as e:
            logger.error(f"Error updating order status: {e}")
    
    async def _create_order(self, symbol: str, side: OrderSide, order_type: OrderType,
                           quantity: float, price: Optional[float], signal: TradeSignal) -> Optional[Order]:
        """Create and submit order to exchange"""
        try:
            start_time = time.perf_counter()
            
            # Generate client order ID
            client_order_id = f"scalper_{uuid.uuid4().hex[:12]}"
            
            # Prepare order parameters
            order_params = {
                'symbol': symbol,
                'side': side.value,
                'type': order_type.value,
                'quantity': quantity,
                'newClientOrderId': client_order_id,
                'timeInForce': 'GTC'  # Good Till Canceled
            }
            
            # Add price for limit orders
            if order_type in [OrderType.LIMIT, OrderType.STOP_LOSS, OrderType.TAKE_PROFIT]:
                order_params['price'] = f"{price:.8f}"
            
            # Add stop price for stop orders
            if order_type in [OrderType.STOP_LOSS, OrderType.TAKE_PROFIT]:
                order_params['stopPrice'] = f"{signal.stop_loss or price:.8f}"
            
            # Submit to exchange
            response = await self.client.create_order(**order_params)
            
            # Record execution latency
            execution_time = (time.perf_counter() - start_time) * 1000  # ms
            self.execution_latency.append(execution_time)
            
            # Create order object
            order = Order(
                id=str(response.get('orderId')),
                client_order_id=client_order_id,
                symbol=symbol,
                side=side,
                order_type=order_type,
                quantity=quantity,
                price=price,
                stop_price=signal.stop_loss,
                status=OrderStatus(response.get('status')),
                timestamp=time.time()
            )
            
            return order
            
        except Exception as e:
            logger.error(f"Error creating order: {e}")
            return None
    
    async def _setup_protection_orders(self, parent_order: Order, signal: TradeSignal):
        """Set up stop loss and take profit orders"""
        try:
            if signal.stop_loss:
                # Create stop loss order
                stop_side = OrderSide.SELL if parent_order.side == OrderSide.BUY else OrderSide.BUY
                
                await self._create_order(
                    symbol=parent_order.symbol,
                    side=stop_side,
                    order_type=OrderType.STOP_LOSS,
                    quantity=parent_order.quantity,
                    price=signal.stop_loss,
                    signal=signal
                )
            
            if signal.take_profit:
                # Create take profit order
                tp_side = OrderSide.SELL if parent_order.side == OrderSide.BUY else OrderSide.BUY
                
                await self._create_order(
                    symbol=parent_order.symbol,
                    side=tp_side,
                    order_type=OrderType.TAKE_PROFIT,
                    quantity=parent_order.quantity,
                    price=signal.take_profit,
                    signal=signal
                )
                
        except Exception as e:
            logger.error(f"Error setting up protection orders: {e}")
    
    def _determine_order_type(self, signal: TradeSignal) -> OrderType:
        """Determine order type based on signal characteristics"""
        if signal.urgency == "URGENT":
            return OrderType.MARKET
        elif signal.price_target:
            return OrderType.LIMIT
        else:
            return OrderType.MARKET
    
    async def _calculate_order_price(self, signal: TradeSignal, order_type: OrderType) -> Optional[float]:
        """Calculate optimal order price"""
        if order_type == OrderType.MARKET:
            return None
        
        if signal.price_target:
            return signal.price_target
        
        # Get current market price
        ticker = await self.client.get_symbol_ticker(symbol=signal.symbol)
        current_price = float(ticker['price'])
        
        # Add small edge for limit orders
        if signal.side == OrderSide.BUY:
            return current_price * 0.9999  # Slightly below market
        else:
            return current_price * 1.0001  # Slightly above market
    
    def _calculate_quantity(self, symbol: str, position_size: float) -> float:
        """Calculate order quantity based on position size"""
        # This would normally use symbol info for proper rounding
        # For now, round to 6 decimal places
        return round(position_size, 6)
    
    async def _check_rate_limits(self) -> bool:
        """Check if we can place another order without exceeding rate limits"""
        current_time = time.time()
        
        # Remove old timestamps
        self.order_timestamps = [
            ts for ts in self.order_timestamps 
            if current_time - ts < 1.0
        ]
        
        # Check if we can place another order
        if len(self.order_timestamps) >= self.max_orders_per_second:
            return False
        
        # Record this order attempt
        self.order_timestamps.append(current_time)
        return True
    
    async def _record_execution_metrics(self, order: Order):
        """Record execution metrics for performance analysis"""
        try:
            # Calculate slippage if we have expected price
            if hasattr(order, 'expected_price') and order.expected_price:
                slippage = abs(order.avg_price - order.expected_price) / order.expected_price
                self.slippage_tracking.append(slippage)
            
            # Log execution
            execution_time = (order.update_time - order.timestamp) * 1000  # ms
            logger.debug(f"Order executed in {execution_time:.2f}ms, "
                        f"avg slippage: {np.mean(self.slippage_tracking[-100:]):.4f}")
            
        except Exception as e:
            logger.error(f"Error recording metrics: {e}")
    
    def get_active_orders(self) -> List[Order]:
        """Get all active orders"""
        return list(self.active_orders.values())
    
    def get_order_history(self, limit: int = 100) -> List[Order]:
        """Get order history"""
        return self.order_history[-limit:]
    
    def get_execution_stats(self) -> Dict[str, float]:
        """Get execution statistics"""
        if not self.execution_latency:
            return {}
        
        return {
            'avg_latency_ms': np.mean(self.execution_latency),
            'p95_latency_ms': np.percentile(self.execution_latency, 95),
            'avg_slippage': np.mean(self.slippage_tracking) if self.slippage_tracking else 0,
            'total_orders': len(self.order_history)
        } 