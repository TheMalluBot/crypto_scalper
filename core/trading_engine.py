"""
Advanced Trading Engine for Crypto Scalping
Similar to Jane Street's quantitative trading systems
"""

import asyncio
import time
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum
import logging
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
from loguru import logger

from ..data.binance_client import BinanceClient
from ..data.data_processor import DataProcessor
from .order_manager import OrderManager
from .position_manager import PositionManager
from .risk_manager import RiskManager
from ..strategies.base_strategy import BaseStrategy
from ..ml.prediction import PredictionEngine
from ..utils.gpu_utils import GPUManager


class TradingState(Enum):
    STOPPED = "stopped"
    STARTING = "starting"
    RUNNING = "running"
    PAUSED = "paused"
    STOPPING = "stopping"
    ERROR = "error"


@dataclass
class TradingMetrics:
    """Real-time trading metrics"""
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    total_pnl: float = 0.0
    unrealized_pnl: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    max_drawdown: float = 0.0
    sharpe_ratio: float = 0.0
    processing_latency: float = 0.0
    orders_per_second: float = 0.0


class TradingEngine:
    """
    High-performance trading engine with GPU acceleration
    Designed for low-latency crypto scalping
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.state = TradingState.STOPPED
        self.metrics = TradingMetrics()
        
        # Initialize components
        self.binance_client = BinanceClient(config['binance'])
        self.data_processor = DataProcessor(config['performance'])
        self.order_manager = OrderManager(self.binance_client)
        self.position_manager = PositionManager()
        self.risk_manager = RiskManager(config['trading'])
        self.prediction_engine = PredictionEngine(config.get('ml', {}))
        self.gpu_manager = GPUManager()
        
        # Strategy management
        self.strategies: Dict[str, BaseStrategy] = {}
        self.active_strategies: List[str] = []
        
        # Performance monitoring
        self.executor = ThreadPoolExecutor(max_workers=8)
        self.last_metrics_update = time.time()
        
        # Data streams
        self.market_data_queue = asyncio.Queue(maxsize=10000)
        self.order_book_queue = asyncio.Queue(maxsize=5000)
        self.trade_queue = asyncio.Queue(maxsize=1000)
        
        logger.info("Trading Engine initialized")
    
    async def start(self):
        """Start the trading engine"""
        if self.state != TradingState.STOPPED:
            raise RuntimeError(f"Cannot start engine in state: {self.state}")
        
        self.state = TradingState.STARTING
        logger.info("Starting trading engine...")
        
        try:
            # Initialize connections
            await self.binance_client.connect()
            await self.gpu_manager.initialize()
            
            # Load and prepare ML models
            await self.prediction_engine.load_models()
            
            # Start data streams
            await self._start_data_streams()
            
            # Start trading loops
            await self._start_trading_loops()
            
            self.state = TradingState.RUNNING
            logger.info("Trading engine started successfully")
            
        except Exception as e:
            self.state = TradingState.ERROR
            logger.error(f"Failed to start trading engine: {e}")
            raise
    
    async def stop(self):
        """Stop the trading engine"""
        if self.state == TradingState.STOPPED:
            return
        
        self.state = TradingState.STOPPING
        logger.info("Stopping trading engine...")
        
        try:
            # Close all positions
            await self.position_manager.close_all_positions()
            
            # Cancel all orders
            await self.order_manager.cancel_all_orders()
            
            # Stop data streams
            await self._stop_data_streams()
            
            # Cleanup resources
            await self.binance_client.disconnect()
            await self.gpu_manager.cleanup()
            
            self.state = TradingState.STOPPED
            logger.info("Trading engine stopped successfully")
            
        except Exception as e:
            logger.error(f"Error stopping trading engine: {e}")
            self.state = TradingState.ERROR
            raise
    
    async def _start_data_streams(self):
        """Start real-time data streams"""
        logger.info("Starting data streams...")
        
        # Market data stream
        asyncio.create_task(self._market_data_stream())
        
        # Order book stream
        asyncio.create_task(self._order_book_stream())
        
        # Trade stream
        asyncio.create_task(self._trade_stream())
        
        # Account updates stream
        asyncio.create_task(self._account_stream())
    
    async def _start_trading_loops(self):
        """Start main trading loops"""
        logger.info("Starting trading loops...")
        
        # Main trading loop
        asyncio.create_task(self._main_trading_loop())
        
        # Risk monitoring loop
        asyncio.create_task(self._risk_monitoring_loop())
        
        # Performance monitoring loop
        asyncio.create_task(self._performance_monitoring_loop())
        
        # ML prediction loop
        asyncio.create_task(self._ml_prediction_loop())
    
    async def _main_trading_loop(self):
        """Main trading loop with microsecond precision"""
        logger.info("Starting main trading loop")
        
        while self.state == TradingState.RUNNING:
            try:
                start_time = time.perf_counter()
                
                # Process market data
                await self._process_market_data()
                
                # Generate trading signals
                signals = await self._generate_signals()
                
                # Execute trades
                if signals:
                    await self._execute_trades(signals)
                
                # Update metrics
                processing_time = time.perf_counter() - start_time
                self.metrics.processing_latency = processing_time * 1000  # ms
                
                # High-frequency loop - aim for < 1ms processing time
                if processing_time < 0.001:
                    await asyncio.sleep(0.001 - processing_time)
                
            except Exception as e:
                logger.error(f"Error in main trading loop: {e}")
                await asyncio.sleep(0.1)
    
    async def _process_market_data(self):
        """Process incoming market data with GPU acceleration"""
        try:
            # Get latest market data
            if not self.market_data_queue.empty():
                data = await self.market_data_queue.get()
                
                # GPU-accelerated processing
                processed_data = await self.data_processor.process_market_data(data)
                
                # Update indicators
                await self._update_indicators(processed_data)
                
        except Exception as e:
            logger.error(f"Error processing market data: {e}")
    
    async def _generate_signals(self):
        """Generate trading signals from active strategies"""
        signals = []
        
        try:
            # Parallel signal generation
            signal_tasks = []
            
            for strategy_name in self.active_strategies:
                strategy = self.strategies.get(strategy_name)
                if strategy:
                    task = asyncio.create_task(strategy.generate_signal())
                    signal_tasks.append(task)
            
            # Wait for all signals
            if signal_tasks:
                results = await asyncio.gather(*signal_tasks, return_exceptions=True)
                
                for result in results:
                    if isinstance(result, Exception):
                        logger.error(f"Strategy error: {result}")
                    elif result:
                        signals.append(result)
        
        except Exception as e:
            logger.error(f"Error generating signals: {e}")
        
        return signals
    
    async def _execute_trades(self, signals: List[Any]):
        """Execute trades based on signals"""
        for signal in signals:
            try:
                # Risk check
                if not await self.risk_manager.check_trade_risk(signal):
                    continue
                
                # Position sizing
                position_size = await self.position_manager.calculate_position_size(signal)
                
                # Execute order
                order = await self.order_manager.place_order(signal, position_size)
                
                if order:
                    await self.position_manager.update_position(order)
                    self.metrics.total_trades += 1
                    
            except Exception as e:
                logger.error(f"Error executing trade: {e}")
    
    async def _risk_monitoring_loop(self):
        """Continuous risk monitoring"""
        while self.state == TradingState.RUNNING:
            try:
                # Check risk limits
                risk_status = await self.risk_manager.check_risk_limits()
                
                if risk_status.get('emergency_stop', False):
                    logger.critical("Emergency stop triggered!")
                    await self.stop()
                    break
                
                await asyncio.sleep(1)  # Check every second
                
            except Exception as e:
                logger.error(f"Error in risk monitoring: {e}")
                await asyncio.sleep(1)
    
    async def _performance_monitoring_loop(self):
        """Monitor and update performance metrics"""
        while self.state == TradingState.RUNNING:
            try:
                # Update metrics
                await self._update_metrics()
                
                # Log performance
                if time.time() - self.last_metrics_update > 60:  # Every minute
                    await self._log_performance()
                    self.last_metrics_update = time.time()
                
                await asyncio.sleep(10)  # Update every 10 seconds
                
            except Exception as e:
                logger.error(f"Error in performance monitoring: {e}")
                await asyncio.sleep(10)
    
    async def _ml_prediction_loop(self):
        """Machine learning prediction loop"""
        while self.state == TradingState.RUNNING:
            try:
                # Generate predictions
                predictions = await self.prediction_engine.predict()
                
                # Update strategy parameters based on predictions
                await self._update_strategy_parameters(predictions)
                
                await asyncio.sleep(5)  # Predict every 5 seconds
                
            except Exception as e:
                logger.error(f"Error in ML prediction: {e}")
                await asyncio.sleep(5)
    
    async def _market_data_stream(self):
        """Real-time market data stream"""
        async for data in self.binance_client.market_data_stream():
            if not self.market_data_queue.full():
                await self.market_data_queue.put(data)
    
    async def _order_book_stream(self):
        """Real-time order book stream"""
        async for data in self.binance_client.order_book_stream():
            if not self.order_book_queue.full():
                await self.order_book_queue.put(data)
    
    async def _trade_stream(self):
        """Real-time trade stream"""
        async for data in self.binance_client.trade_stream():
            if not self.trade_queue.full():
                await self.trade_queue.put(data)
    
    async def _account_stream(self):
        """Real-time account updates stream"""
        async for data in self.binance_client.account_stream():
            await self.position_manager.update_from_account_data(data)
    
    async def _update_indicators(self, data):
        """Update technical indicators"""
        # Placeholder for indicator updates
        pass
    
    async def _update_metrics(self):
        """Update trading metrics"""
        # Calculate win rate
        if self.metrics.total_trades > 0:
            self.metrics.win_rate = self.metrics.winning_trades / self.metrics.total_trades
        
        # Calculate profit factor
        # Implementation depends on trade tracking
        pass
    
    async def _log_performance(self):
        """Log performance metrics"""
        logger.info(f"Trading Performance: Trades={self.metrics.total_trades}, "
                   f"Win Rate={self.metrics.win_rate:.2%}, "
                   f"PnL={self.metrics.total_pnl:.2f}, "
                   f"Latency={self.metrics.processing_latency:.2f}ms")
    
    async def _update_strategy_parameters(self, predictions):
        """Update strategy parameters based on ML predictions"""
        # Placeholder for strategy parameter updates
        pass
    
    async def _stop_data_streams(self):
        """Stop data streams"""
        # Implementation for stopping streams
        pass
    
    def add_strategy(self, name: str, strategy: BaseStrategy):
        """Add a trading strategy"""
        self.strategies[name] = strategy
        logger.info(f"Added strategy: {name}")
    
    def activate_strategy(self, name: str):
        """Activate a trading strategy"""
        if name in self.strategies and name not in self.active_strategies:
            self.active_strategies.append(name)
            logger.info(f"Activated strategy: {name}")
    
    def deactivate_strategy(self, name: str):
        """Deactivate a trading strategy"""
        if name in self.active_strategies:
            self.active_strategies.remove(name)
            logger.info(f"Deactivated strategy: {name}")
    
    def get_metrics(self) -> TradingMetrics:
        """Get current trading metrics"""
        return self.metrics
    
    def get_state(self) -> TradingState:
        """Get current trading state"""
        return self.state 