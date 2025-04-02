import torch
import numpy as np
import pandas as pd
import time
from binance.client import Client  # 用于获取实时市场数据
from train2 import DQN, DQNAgent

# Binance API 配置 (需要替换你的 API KEY)
# API_KEY = "your_api_key"
# API_SECRET = "your_api_secret"
client = Client()

# 交易参数
TRADING_PAIR = "BNBUSDT"  # 交易对
INITIAL_BALANCE = 10000  # 初始资金
crypto_held = 0  # 持仓
balance = INITIAL_BALANCE  # 账户余额
last_price = 0  # 记录上次买入的价格

# **加载训练好的模型**
agent = DQNAgent(state_dim=5, action_dim=3)
agent.model.load_state_dict(torch.load("models/dqn_20250328_153409_episodes_1000_epsilon_0.01.pth"))
agent.model.eval()  # 设为评估模式


# **获取实时市场数据**
def get_latest_data():
    """ 获取最近的一根 1h K 线数据 """
    klines = client.get_klines(symbol=TRADING_PAIR, interval=Client.KLINE_INTERVAL_1HOUR, limit=2)
    latest_kline = klines[-1]
    return {
        "open": float(latest_kline[1]),
        "high": float(latest_kline[2]),
        "low": float(latest_kline[3]),
        "close": float(latest_kline[4])
    }


# **执行交易**
def trade():
    global balance, crypto_held, last_price

    # **获取最新市场数据**
    market_data = get_latest_data()
    state = np.array([
        market_data["open"],
        market_data["high"],
        market_data["low"],
        market_data["close"],
        crypto_held
    ], dtype=np.float32)

    # **用 DQN 模型选择动作**
    action = agent.select_action(state)
    current_price = market_data["close"]

    # **执行交易逻辑**
    if action == 1 and balance >= current_price:  # **买入**
        crypto_held += 1
        balance -= current_price
        last_price = current_price
        print(f"买入 1 BNB, 价格: {current_price:.2f}, 账户余额: {balance:.2f}, 持仓: {crypto_held}")

    elif action == 2 and crypto_held > 0:  # **卖出**
        crypto_held -= 1
        balance += current_price
        profit = current_price - last_price
        print(
            f"卖出 1 BNB, 价格: {current_price:.2f}, 盈利: {profit:.2f}, 账户余额: {balance:.2f}, 持仓: {crypto_held}")

    else:
        print(f"保持持仓, 价格: {current_price:.2f}, 账户余额: {balance:.2f}, 持仓: {crypto_held}")

    # **记录交易数据**
    trade_log = pd.DataFrame([{
        "time": pd.Timestamp.now(),
        "action": action,
        "price": current_price,
        "balance": balance,
        "crypto_held": crypto_held
    }])
    trade_log.to_csv("trading_log.csv", mode="a", header=False, index=False)


# **每小时运行一次**
if __name__ == "__main__":
    while True:
        trade()
        print("=" * 40)
        time.sleep(3600)  # 每小时执行一次
