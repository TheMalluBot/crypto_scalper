import streamlit as st
import yaml

st.set_page_config(page_title="Crypto Scalper Dashboard", layout="wide")
st.title("🚀 Crypto Scalper Trading Dashboard")

st.markdown("---")

# Helper to load YAML config
def load_yaml_config(path):
    try:
        with open(path, 'r') as f:
            return yaml.safe_load(f)
    except Exception as e:
        return {"error": str(e)}

# Load configs
trading_config = load_yaml_config("config/trading_config.yaml")
binance_config = load_yaml_config("config/binance_config.yaml")
strategies_config = load_yaml_config("config/strategies_config.yaml")

# Layout
col1, col2, col3 = st.columns(3)

with col1:
    st.header("Trading Config")
    st.json(trading_config)

with col2:
    st.header("Binance Config")
    st.json(binance_config)

with col3:
    st.header("Strategies Config")
    st.json(strategies_config)

st.markdown("---")
st.info("This is a basic GUI. You can expand it to show live stats, trades, and more!") 