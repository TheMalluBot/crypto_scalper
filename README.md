# Crypto Scalper - Professional Algorithmic Trading System

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Binance](https://img.shields.io/badge/Exchange-Binance-yellow.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

A professional-grade, high-frequency cryptocurrency scalping system designed for institutional-level trading performance. Built with advanced risk management, GPU acceleration, and multiple trading strategies.

## 🚀 Features

### Core Trading Engine
- **Microsecond Latency**: Sub-millisecond order execution
- **GPU Acceleration**: CUDA/OpenCL support for technical indicators
- **Real-time Data Processing**: WebSocket streams for market data
- **Advanced Risk Management**: Multi-layer protection with emergency stops
- **Portfolio Management**: Kelly Criterion position sizing

### Trading Strategies
- **Micro Scalping**: Ultra-short term price movement capture
- **Order Book Imbalance**: Institutional-grade flow analysis
- **ML Prediction**: Machine learning price forecasting
- **Momentum Scalping**: Trend-following scalping
- **Statistical Arbitrage**: Mean reversion strategies

### Risk Management
- **Position Sizing**: Kelly Criterion and risk-based sizing
- **Stop Loss/Take Profit**: Automated risk controls
- **Drawdown Protection**: Maximum drawdown limits
- **Daily Loss Limits**: Risk-adjusted trading limits
- **Emergency Stops**: Market volatility protection

### Performance Analytics
- **Real-time Metrics**: Live P&L, Sharpe ratio, win rate
- **Performance Dashboard**: Rich terminal interface
- **Trade Analytics**: Detailed execution metrics
- **Risk Reporting**: VaR and risk-adjusted returns

## 📁 Project Structure

```
src/crypto_scalper/
├── config/                 # Configuration files
│   ├── trading_config.yaml    # Trading parameters
│   ├── binance_config.yaml    # Binance API settings
│   └── strategies_config.yaml # Strategy configurations
├── core/                   # Core trading components
│   ├── trading_engine.py      # Main trading engine
│   ├── order_manager.py       # Order execution
│   ├── position_manager.py    # Position tracking
│   └── risk_manager.py        # Risk controls
├── data/                   # Data processing
│   ├── binance_client.py      # Binance integration
│   └── data_processor.py      # Market data processing
├── strategies/             # Trading strategies
│   ├── base_strategy.py       # Strategy framework
│   ├── scalping_strategies.py # Scalping implementations
│   └── ml_strategies.py       # ML-based strategies
├── utils.py               # Utility functions
├── main.py               # Application entry point
└── requirements.txt      # Python dependencies
```

## 🛠️ Installation

### Requirements
- Python 3.8+
- Windows 10/11 (tested)
- GPU with CUDA support (recommended)
- Binance account with API access

### Quick Start

1. **Clone and Install Dependencies**
```bash
cd src/crypto_scalper
pip install -r requirements.txt
```

2. **Configure API Keys**
```bash
# Edit config files with your Binance API credentials
notepad config/binance_config.yaml
```

3. **Run the System**
```bash
# Run with live dashboard (testnet mode)
python -m main run --testnet

# Run without dashboard
python -m main run --dashboard false

# Validate configuration
python -m main validate-config
```

## ⚙️ Configuration

### Trading Configuration (`trading_config.yaml`)
```yaml
trading:
  max_position_size: 0.02      # 2% of portfolio per trade
  max_daily_loss: 0.05         # 5% daily loss limit
  max_drawdown: 0.10           # 10% max drawdown
  stop_loss_pct: 0.005         # 0.5% stop loss
  take_profit_pct: 0.015       # 1.5% take profit
  risk_reward_ratio: 3.0       # Minimum 1:3 risk/reward

performance:
  use_gpu: true                # Enable GPU acceleration
  gpu_memory_fraction: 0.8     # GPU memory usage
  cpu_cores: -1                # Use all CPU cores
```

### Strategy Configuration (`strategies_config.yaml`)
```yaml
strategies:
  micro_scalping:
    enabled: true
    timeframe: "1s"
    profit_target: 0.003        # 0.3% target
    signal_cooldown: 1.0        # 1 second between signals
    
  orderbook_imbalance:
    enabled: true
    imbalance_threshold: 0.7    # 70% imbalance trigger
    depth_levels: 10            # Order book depth
    
  ml_prediction:
    enabled: false              # Requires trained models
    model_type: "LSTM"
    confidence_threshold: 0.75  # 75% confidence minimum
```

## 🎯 Trading Strategies

### 1. Micro Scalping
- **Objective**: Capture small price movements (0.1-0.5%)
- **Timeframe**: 1-5 seconds
- **Indicators**: EMA crossovers, RSI, volume analysis
- **Risk**: Tight stop losses (0.1-0.2%)

### 2. Order Book Imbalance
- **Objective**: Detect institutional flow
- **Method**: Analyze bid/ask volume imbalances
- **Trigger**: 70%+ imbalance threshold
- **Speed**: Ultra-low latency execution

### 3. ML Prediction
- **Objective**: Predict short-term price movements
- **Models**: LSTM, XGBoost, ensemble methods
- **Features**: Technical indicators, order book data
- **Confidence**: 75%+ prediction confidence

## 📊 Risk Management

### Position Sizing
- **Kelly Criterion**: Optimal position sizing
- **Maximum Risk**: 2% of portfolio per trade
- **Portfolio Heat**: Maximum 10% total exposure

### Stop Loss Strategy
- **Dynamic Stops**: ATR-based stop levels
- **Emergency Stops**: Market volatility protection
- **Trailing Stops**: Profit protection mechanism

### Risk Limits
- **Daily Loss**: 5% maximum daily loss
- **Drawdown**: 10% maximum drawdown
- **Volatility**: Dynamic exposure adjustment

## 🎮 Live Dashboard

The system includes a beautiful terminal dashboard showing:

- **Real-time P&L**: Live profit/loss tracking
- **Strategy Performance**: Individual strategy metrics
- **Open Positions**: Current trades and status
- **Risk Metrics**: Drawdown, Sharpe ratio, VaR
- **Order Flow**: Recent trade execution

## 🔧 Advanced Features

### GPU Acceleration
```python
# Automatic GPU detection and usage
performance:
  use_gpu: true
  gpu_memory_fraction: 0.8
```

### High-Frequency Features
- Sub-millisecond order execution
- Real-time market data processing
- Parallel signal generation
- Low-latency networking

### Machine Learning
- Online model training
- Feature engineering
- Model ensemble methods
- Prediction confidence scoring

## 📈 Performance Optimization

### Latency Optimization
- Async/await architecture
- Connection pooling
- Memory pre-allocation
- JIT compilation with Numba

### Resource Management
- GPU memory optimization
- CPU core utilization
- Network bandwidth management
- Memory-mapped files

## 🛡️ Security

- **API Key Security**: Encrypted credential storage
- **Rate Limiting**: Binance API compliance
- **Error Handling**: Robust exception management
- **Logging**: Comprehensive audit trails

## 📊 Backtesting (Coming Soon)

```bash
# Run historical backtests
python -m main backtest --symbol BTCUSDT --days 30
```

## 🚨 Important Notes

### Legal Disclaimer
- This software is for educational purposes
- Trading involves substantial risk of loss
- Past performance does not guarantee future results
- Use at your own risk

### Testnet Mode
- Always test with Binance testnet first
- Validate strategies before live trading
- Monitor risk limits carefully

### API Requirements
- Binance account with spot trading enabled
- API key with trading permissions
- IP whitelist (recommended)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Implement your changes
4. Add tests and documentation
5. Submit a pull request

## 📞 Support

For questions and support:
- Review the documentation
- Check configuration examples
- Test with small position sizes
- Use testnet for development

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

⚡ **Built for Speed, Designed for Profit** ⚡

*Professional cryptocurrency scalping system with institutional-grade performance* 