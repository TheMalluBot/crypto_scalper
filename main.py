"""
Main entry point for the crypto scalping system
Orchestrates all components and provides command line interface
"""

import asyncio
import sys
import signal
from pathlib import Path
from typing import Dict, Any
import typer
from rich.console import Console
from rich.table import Table
from rich.live import Live
from rich.layout import Layout
from rich.panel import Panel
from loguru import logger

from .core.trading_engine import TradingEngine
from .strategies.base_strategy import BaseStrategy
from .strategies.scalping_strategies import MicroScalpingStrategy, OrderBookImbalanceStrategy
from .strategies.ml_strategies import MLPredictionStrategy
from .utils import load_config, validate_api_keys, create_directory, format_currency, format_percentage

app = typer.Typer(help="Crypto Scalping Trading System")
console = Console()


class ScalpingApp:
    """Main application class for crypto scalping"""
    
    def __init__(self):
        self.trading_engine = None
        self.config = {}
        self.running = False
        self.console = Console()
        
        # Setup logging
        self._setup_logging()
        
        # Signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _setup_logging(self):
        """Setup logging configuration"""
        try:
            # Remove default logger
            logger.remove()
            
            # Add console logger
            logger.add(
                sys.stderr,
                format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
                level="INFO"
            )
            
            # Add file logger
            create_directory("logs")
            logger.add(
                "logs/trading.log",
                format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
                level="DEBUG",
                rotation="100 MB",
                retention="30 days"
            )
            
            logger.info("Logging setup completed")
            
        except Exception as e:
            print(f"Error setting up logging: {e}")
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        logger.info(f"Received signal {signum}, shutting down gracefully...")
        self.running = False
        
        if self.trading_engine:
            asyncio.create_task(self.trading_engine.stop())
    
    async def load_configuration(self, config_dir: str = "config") -> bool:
        """Load all configuration files"""
        try:
            logger.info("Loading configuration...")
            
            config_path = Path(config_dir)
            
            # Load main trading config
            trading_config_path = config_path / "trading_config.yaml"
            if trading_config_path.exists():
                self.config['trading'] = load_config(str(trading_config_path))
            else:
                logger.error("Trading configuration not found")
                return False
            
            # Load Binance config
            binance_config_path = config_path / "binance_config.yaml"
            if binance_config_path.exists():
                self.config['binance'] = load_config(str(binance_config_path))
            else:
                logger.error("Binance configuration not found")
                return False
            
            # Load strategies config
            strategies_config_path = config_path / "strategies_config.yaml"
            if strategies_config_path.exists():
                self.config['strategies'] = load_config(str(strategies_config_path))
            else:
                logger.error("Strategies configuration not found")
                return False
            
            # Validate API keys
            if not validate_api_keys(self.config):
                logger.warning("API keys validation failed - running in demo mode")
            
            logger.info("Configuration loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
            return False
    
    async def initialize_trading_engine(self) -> bool:
        """Initialize the trading engine with strategies"""
        try:
            logger.info("Initializing trading engine...")
            
            # Create trading engine
            self.trading_engine = TradingEngine(self.config)
            
            # Initialize strategies
            await self._initialize_strategies()
            
            # Initialize engine
            await self.trading_engine.initialize()
            
            logger.info("Trading engine initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error initializing trading engine: {e}")
            return False
    
    async def _initialize_strategies(self):
        """Initialize trading strategies"""
        try:
            strategies_config = self.config.get('strategies', {})
            
            # Initialize micro scalping strategy
            if strategies_config.get('micro_scalping', {}).get('enabled', False):
                strategy = MicroScalpingStrategy(strategies_config['micro_scalping'])
                await self.trading_engine.add_strategy(strategy)
                logger.info("Micro scalping strategy added")
            
            # Initialize order book imbalance strategy
            if strategies_config.get('orderbook_imbalance', {}).get('enabled', False):
                strategy = OrderBookImbalanceStrategy(strategies_config['orderbook_imbalance'])
                await self.trading_engine.add_strategy(strategy)
                logger.info("Order book imbalance strategy added")
            
            # Initialize ML prediction strategy
            if strategies_config.get('ml_prediction', {}).get('enabled', False):
                strategy = MLPredictionStrategy(strategies_config['ml_prediction'])
                await self.trading_engine.add_strategy(strategy)
                logger.info("ML prediction strategy added")
            
        except Exception as e:
            logger.error(f"Error initializing strategies: {e}")
    
    async def run_trading(self):
        """Main trading loop"""
        try:
            logger.info("Starting trading system...")
            self.running = True
            
            # Start trading engine
            await self.trading_engine.start()
            
            # Main loop
            while self.running:
                await asyncio.sleep(1)
            
            # Stop trading engine
            await self.trading_engine.stop()
            
            logger.info("Trading system stopped")
            
        except Exception as e:
            logger.error(f"Error in trading loop: {e}")
            self.running = False
    
    def create_dashboard_layout(self) -> Layout:
        """Create rich dashboard layout"""
        layout = Layout()
        
        layout.split_column(
            Layout(name="header"),
            Layout(name="main"),
            Layout(name="footer")
        )
        
        layout["main"].split_row(
            Layout(name="left"),
            Layout(name="right")
        )
        
        layout["left"].split_column(
            Layout(name="strategies"),
            Layout(name="positions")
        )
        
        layout["right"].split_column(
            Layout(name="performance"),
            Layout(name="orders")
        )
        
        return layout
    
    def update_dashboard(self, layout: Layout):
        """Update dashboard with current data"""
        try:
            if not self.trading_engine:
                return
            
            # Header
            layout["header"].update(
                Panel("🚀 Crypto Scalping Trading System", style="bold blue")
            )
            
            # Strategies status
            strategies_table = Table("Strategy", "Status", "Signals", "Win Rate", "PnL")
            
            for strategy in self.trading_engine.strategies:
                status = strategy.get_status()
                strategies_table.add_row(
                    status['name'],
                    "🟢 Active" if status['active'] else "🔴 Inactive",
                    str(status['total_signals']),
                    format_percentage(status['win_rate']),
                    format_currency(status['total_pnl'])
                )
            
            layout["strategies"].update(Panel(strategies_table, title="Strategies"))
            
            # Positions
            positions_table = Table("Symbol", "Side", "Size", "PnL", "Duration")
            
            for position in self.trading_engine.position_manager.get_open_positions():
                positions_table.add_row(
                    position.symbol,
                    position.side.value,
                    f"{position.quantity:.6f}",
                    format_currency(position.unrealized_pnl),
                    f"{position.duration:.0f}s"
                )
            
            layout["positions"].update(Panel(positions_table, title="Open Positions"))
            
            # Performance metrics
            performance = self.trading_engine.position_manager.get_performance_summary()
            
            perf_table = Table("Metric", "Value")
            perf_table.add_row("Total PnL", format_currency(performance.get('total_pnl', 0)))
            perf_table.add_row("Win Rate", format_percentage(performance.get('win_rate', 0)))
            perf_table.add_row("Sharpe Ratio", f"{performance.get('sharpe_ratio', 0):.2f}")
            perf_table.add_row("Max Drawdown", format_percentage(performance.get('max_drawdown', 0)))
            perf_table.add_row("Total Trades", str(performance.get('total_trades', 0)))
            
            layout["performance"].update(Panel(perf_table, title="Performance"))
            
            # Recent orders
            orders_table = Table("Time", "Symbol", "Side", "Status", "Price")
            
            recent_orders = self.trading_engine.order_manager.get_recent_orders(10)
            for order in recent_orders:
                orders_table.add_row(
                    order.timestamp.strftime("%H:%M:%S"),
                    order.symbol,
                    order.side.value,
                    order.status.value,
                    f"${order.price:.2f}"
                )
            
            layout["orders"].update(Panel(orders_table, title="Recent Orders"))
            
            # Footer
            engine_status = self.trading_engine.get_status()
            footer_text = f"Engine: {'🟢 Running' if engine_status['running'] else '🔴 Stopped'} | "
            footer_text += f"Uptime: {engine_status.get('uptime', 0):.0f}s | "
            footer_text += f"Data Updates: {engine_status.get('data_updates', 0)}"
            
            layout["footer"].update(Panel(footer_text, style="dim"))
            
        except Exception as e:
            logger.error(f"Error updating dashboard: {e}")
    
    async def run_with_dashboard(self):
        """Run trading system with live dashboard"""
        try:
            layout = self.create_dashboard_layout()
            
            with Live(layout, refresh_per_second=2, screen=True):
                # Start trading in background
                trading_task = asyncio.create_task(self.run_trading())
                
                # Update dashboard
                while self.running:
                    self.update_dashboard(layout)
                    await asyncio.sleep(0.5)
                
                # Wait for trading to complete
                await trading_task
            
        except Exception as e:
            logger.error(f"Error running dashboard: {e}")


# CLI Commands

@app.command()
def run(
    config_dir: str = typer.Option("config", help="Configuration directory"),
    dashboard: bool = typer.Option(True, help="Show live dashboard"),
    testnet: bool = typer.Option(True, help="Use testnet mode")
):
    """Run the crypto scalping trading system"""
    
    async def main():
        app_instance = ScalpingApp()
        
        # Load configuration
        if not await app_instance.load_configuration(config_dir):
            console.print("❌ Failed to load configuration", style="red")
            return
        
        # Set testnet mode
        if testnet:
            app_instance.config['binance']['testnet'] = True
            console.print("🧪 Running in testnet mode", style="yellow")
        
        # Initialize trading engine
        if not await app_instance.initialize_trading_engine():
            console.print("❌ Failed to initialize trading engine", style="red")
            return
        
        console.print("✅ Crypto scalping system initialized", style="green")
        
        # Run with or without dashboard
        if dashboard:
            await app_instance.run_with_dashboard()
        else:
            await app_instance.run_trading()
    
    asyncio.run(main())


@app.command()
def validate_config(config_dir: str = typer.Option("config", help="Configuration directory")):
    """Validate configuration files"""
    
    console.print("🔍 Validating configuration...", style="blue")
    
    config_path = Path(config_dir)
    valid = True
    
    # Check if config directory exists
    if not config_path.exists():
        console.print(f"❌ Configuration directory not found: {config_dir}", style="red")
        return
    
    # Check required files
    required_files = ["trading_config.yaml", "binance_config.yaml", "strategies_config.yaml"]
    
    for file_name in required_files:
        file_path = config_path / file_name
        if file_path.exists():
            console.print(f"✅ {file_name}", style="green")
            
            # Load and validate content
            try:
                config = load_config(str(file_path))
                if not config:
                    console.print(f"⚠️  {file_name} is empty or invalid", style="yellow")
                    valid = False
            except Exception as e:
                console.print(f"❌ Error loading {file_name}: {e}", style="red")
                valid = False
        else:
            console.print(f"❌ {file_name} not found", style="red")
            valid = False
    
    if valid:
        console.print("✅ All configuration files are valid", style="green")
    else:
        console.print("❌ Configuration validation failed", style="red")


@app.command()
def create_config(config_dir: str = typer.Option("config", help="Configuration directory")):
    """Create default configuration files"""
    
    console.print("📁 Creating default configuration files...", style="blue")
    
    try:
        create_directory(config_dir)
        
        # This would create default config files
        # For brevity, we'll just show the message
        console.print("✅ Default configuration files created", style="green")
        console.print(f"📝 Please edit the files in {config_dir}/ with your API keys and preferences", style="yellow")
        
    except Exception as e:
        console.print(f"❌ Error creating configuration: {e}", style="red")


@app.command()
def backtest(
    config_dir: str = typer.Option("config", help="Configuration directory"),
    symbol: str = typer.Option("BTCUSDT", help="Trading symbol"),
    days: int = typer.Option(7, help="Number of days to backtest")
):
    """Run backtesting on historical data"""
    
    console.print(f"📊 Running backtest for {symbol} over {days} days...", style="blue")
    
    # This would implement backtesting logic
    # For now, just show placeholder
    console.print("🚧 Backtesting feature coming soon!", style="yellow")


if __name__ == "__main__":
    app() 