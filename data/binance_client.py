"""
Binance API Client for crypto scalping
Handles REST API calls and WebSocket streams
"""

import asyncio
import json
import time
from typing import Dict, List, Optional, Any, AsyncGenerator
import hmac
import hashlib
import urllib.parse

import aiohttp
import websockets
from loguru import logger

from binance.client import Client
from binance.streams import BinanceSocketManager


class BinanceClient:
    """
    High-performance Binance API client
    Supports both REST API and WebSocket streams
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.api_key = config.get('api_key', '')
        self.api_secret = config.get('api_secret', '')
        self.testnet = config.get('testnet', True)
        self.timeout = config.get('timeout', 30)
        
        # Initialize Binance client
        self.client = Client(
            api_key=self.api_key,
            api_secret=self.api_secret,
            testnet=self.testnet
        )
        
        # WebSocket connections
        self.socket_manager = None
        self.websocket_connections = {}
        
        # Rate limiting
        self.request_timestamps = []
        self.max_requests_per_minute = config.get('requests_per_minute', 1200)
        
        # Connection status
        self.connected = False
        
        logger.info(f"Binance Client initialized (testnet: {self.testnet})")
    
    async def connect(self):
        """Initialize connections"""
        try:
            # Test connection
            server_time = self.client.get_server_time()
            logger.info(f"Connected to Binance (server time: {server_time})")
            
            # Initialize WebSocket manager
            self.socket_manager = BinanceSocketManager(self.client)
            
            self.connected = True
            
        except Exception as e:
            logger.error(f"Failed to connect to Binance: {e}")
            raise
    
    async def disconnect(self):
        """Close all connections"""
        try:
            if self.socket_manager:
                await self.socket_manager.close()
            
            # Close any active WebSocket connections
            for conn in self.websocket_connections.values():
                if not conn.closed:
                    await conn.close()
            
            self.connected = False
            logger.info("Disconnected from Binance")
            
        except Exception as e:
            logger.error(f"Error disconnecting: {e}")
    
    # REST API Methods
    async def get_symbol_ticker(self, symbol: str) -> Dict[str, Any]:
        """Get symbol ticker price"""
        try:
            if not await self._check_rate_limit():
                await asyncio.sleep(0.1)
            
            ticker = self.client.get_symbol_ticker(symbol=symbol)
            return ticker
            
        except Exception as e:
            logger.error(f"Error getting ticker for {symbol}: {e}")
            return {}
    
    async def get_orderbook(self, symbol: str, limit: int = 100) -> Dict[str, Any]:
        """Get order book"""
        try:
            if not await self._check_rate_limit():
                await asyncio.sleep(0.1)
            
            orderbook = self.client.get_order_book(symbol=symbol, limit=limit)
            return orderbook
            
        except Exception as e:
            logger.error(f"Error getting orderbook for {symbol}: {e}")
            return {}
    
    async def get_klines(self, symbol: str, interval: str, limit: int = 500) -> List[List]:
        """Get candlestick data"""
        try:
            if not await self._check_rate_limit():
                await asyncio.sleep(0.1)
            
            klines = self.client.get_klines(
                symbol=symbol,
                interval=interval,
                limit=limit
            )
            return klines
            
        except Exception as e:
            logger.error(f"Error getting klines for {symbol}: {e}")
            return []
    
    async def create_order(self, **kwargs) -> Dict[str, Any]:
        """Create a new order"""
        try:
            if not await self._check_rate_limit():
                await asyncio.sleep(0.1)
            
            order = self.client.create_order(**kwargs)
            logger.info(f"Order created: {order.get('orderId')}")
            return order
            
        except Exception as e:
            logger.error(f"Error creating order: {e}")
            raise
    
    async def cancel_order(self, symbol: str, **kwargs) -> Dict[str, Any]:
        """Cancel an order"""
        try:
            if not await self._check_rate_limit():
                await asyncio.sleep(0.1)
            
            result = self.client.cancel_order(symbol=symbol, **kwargs)
            return result
            
        except Exception as e:
            logger.error(f"Error canceling order: {e}")
            return {}
    
    async def get_account(self) -> Dict[str, Any]:
        """Get account information"""
        try:
            if not await self._check_rate_limit():
                await asyncio.sleep(0.1)
            
            account = self.client.get_account()
            return account
            
        except Exception as e:
            logger.error(f"Error getting account info: {e}")
            return {}
    
    async def get_open_orders(self, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get open orders"""
        try:
            if not await self._check_rate_limit():
                await asyncio.sleep(0.1)
            
            orders = self.client.get_open_orders(symbol=symbol)
            return orders
            
        except Exception as e:
            logger.error(f"Error getting open orders: {e}")
            return []
    
    # WebSocket Streams
    async def market_data_stream(self, symbols: Optional[List[str]] = None) -> AsyncGenerator[Dict[str, Any], None]:
        """Real-time market data stream"""
        if not symbols:
            symbols = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT']
        
        try:
            # Create ticker stream
            stream_name = f"{'|'.join([s.lower() + '@ticker' for s in symbols])}"
            
            async with websockets.connect(
                f"wss://stream.binance.com:9443/ws/{stream_name}"
            ) as websocket:
                self.websocket_connections['market_data'] = websocket
                
                async for message in websocket:
                    try:
                        data = json.loads(message)
                        
                        # Process ticker data
                        if 'stream' in data:
                            yield self._process_ticker_data(data['data'])
                        else:
                            yield self._process_ticker_data(data)
                            
                    except json.JSONDecodeError:
                        logger.warning("Invalid JSON in market data stream")
                        continue
                    except Exception as e:
                        logger.error(f"Error processing market data: {e}")
                        continue
                        
        except Exception as e:
            logger.error(f"Market data stream error: {e}")
    
    async def order_book_stream(self, symbols: Optional[List[str]] = None) -> AsyncGenerator[Dict[str, Any], None]:
        """Real-time order book stream"""
        if not symbols:
            symbols = ['BTCUSDT', 'ETHUSDT']
        
        try:
            # Create depth stream
            stream_name = f"{'|'.join([s.lower() + '@depth@100ms' for s in symbols])}"
            
            async with websockets.connect(
                f"wss://stream.binance.com:9443/ws/{stream_name}"
            ) as websocket:
                self.websocket_connections['orderbook'] = websocket
                
                async for message in websocket:
                    try:
                        data = json.loads(message)
                        
                        if 'stream' in data:
                            yield self._process_orderbook_data(data['data'])
                        else:
                            yield self._process_orderbook_data(data)
                            
                    except json.JSONDecodeError:
                        logger.warning("Invalid JSON in orderbook stream")
                        continue
                    except Exception as e:
                        logger.error(f"Error processing orderbook data: {e}")
                        continue
                        
        except Exception as e:
            logger.error(f"Order book stream error: {e}")
    
    async def trade_stream(self, symbols: Optional[List[str]] = None) -> AsyncGenerator[Dict[str, Any], None]:
        """Real-time trade stream"""
        if not symbols:
            symbols = ['BTCUSDT', 'ETHUSDT']
        
        try:
            # Create trade stream
            stream_name = f"{'|'.join([s.lower() + '@trade' for s in symbols])}"
            
            async with websockets.connect(
                f"wss://stream.binance.com:9443/ws/{stream_name}"
            ) as websocket:
                self.websocket_connections['trades'] = websocket
                
                async for message in websocket:
                    try:
                        data = json.loads(message)
                        
                        if 'stream' in data:
                            yield self._process_trade_data(data['data'])
                        else:
                            yield self._process_trade_data(data)
                            
                    except json.JSONDecodeError:
                        logger.warning("Invalid JSON in trade stream")
                        continue
                    except Exception as e:
                        logger.error(f"Error processing trade data: {e}")
                        continue
                        
        except Exception as e:
            logger.error(f"Trade stream error: {e}")
    
    async def account_stream(self) -> AsyncGenerator[Dict[str, Any], None]:
        """Real-time account updates stream"""
        try:
            # Get listen key for user data stream
            listen_key_response = self.client.stream_get_listen_key()
            listen_key = listen_key_response['listenKey']
            
            # Connect to user data stream
            async with websockets.connect(
                f"wss://stream.binance.com:9443/ws/{listen_key}"
            ) as websocket:
                self.websocket_connections['account'] = websocket
                
                # Keep-alive task for listen key
                keep_alive_task = asyncio.create_task(
                    self._keep_alive_listen_key(listen_key)
                )
                
                try:
                    async for message in websocket:
                        try:
                            data = json.loads(message)
                            yield self._process_account_data(data)
                            
                        except json.JSONDecodeError:
                            logger.warning("Invalid JSON in account stream")
                            continue
                        except Exception as e:
                            logger.error(f"Error processing account data: {e}")
                            continue
                finally:
                    keep_alive_task.cancel()
                    
        except Exception as e:
            logger.error(f"Account stream error: {e}")
    
    # Data Processing Methods
    def _process_ticker_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process ticker data"""
        return {
            'type': 'ticker',
            'symbol': data.get('s'),
            'price': float(data.get('c', 0)),
            'price_change': float(data.get('P', 0)),
            'volume': float(data.get('v', 0)),
            'high': float(data.get('h', 0)),
            'low': float(data.get('l', 0)),
            'timestamp': data.get('E', time.time() * 1000)
        }
    
    def _process_orderbook_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process order book data"""
        return {
            'type': 'orderbook',
            'symbol': data.get('s'),
            'bids': [[float(bid[0]), float(bid[1])] for bid in data.get('b', [])],
            'asks': [[float(ask[0]), float(ask[1])] for ask in data.get('a', [])],
            'timestamp': data.get('E', time.time() * 1000)
        }
    
    def _process_trade_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process trade data"""
        return {
            'type': 'trade',
            'symbol': data.get('s'),
            'price': float(data.get('p', 0)),
            'quantity': float(data.get('q', 0)),
            'side': 'buy' if data.get('m', False) else 'sell',
            'timestamp': data.get('T', time.time() * 1000)
        }
    
    def _process_account_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process account update data"""
        event_type = data.get('e')
        
        if event_type == 'executionReport':
            # Order update
            return {
                'type': 'order_update',
                'symbol': data.get('s'),
                'order_id': data.get('i'),
                'client_order_id': data.get('c'),
                'status': data.get('X'),
                'side': data.get('S'),
                'order_type': data.get('o'),
                'filled_qty': float(data.get('z', 0)),
                'price': float(data.get('p', 0)),
                'timestamp': data.get('T', time.time() * 1000)
            }
        elif event_type == 'balanceUpdate':
            # Balance update
            return {
                'type': 'balance_update',
                'asset': data.get('a'),
                'balance': float(data.get('d', 0)),
                'timestamp': data.get('T', time.time() * 1000)
            }
        else:
            return {
                'type': 'unknown',
                'data': data,
                'timestamp': time.time() * 1000
            }
    
    async def _keep_alive_listen_key(self, listen_key: str):
        """Keep user data stream alive"""
        while True:
            try:
                await asyncio.sleep(1800)  # 30 minutes
                self.client.stream_keepalive(listenKey=listen_key)
                logger.debug("Listen key keep-alive sent")
            except Exception as e:
                logger.error(f"Error keeping listen key alive: {e}")
                break
    
    async def _check_rate_limit(self) -> bool:
        """Check if we can make another request"""
        current_time = time.time()
        
        # Remove timestamps older than 1 minute
        self.request_timestamps = [
            ts for ts in self.request_timestamps
            if current_time - ts < 60
        ]
        
        # Check if we're within limits
        if len(self.request_timestamps) >= self.max_requests_per_minute:
            return False
        
        # Record this request
        self.request_timestamps.append(current_time)
        return True
    
    def get_connection_status(self) -> Dict[str, Any]:
        """Get connection status"""
        return {
            'connected': self.connected,
            'websocket_connections': list(self.websocket_connections.keys()),
            'requests_per_minute': len(self.request_timestamps),
            'testnet': self.testnet
        } 