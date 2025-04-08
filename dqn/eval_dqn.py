#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：CryptoFinance 
@File    ：eval_dqn_backtest.py
@Author  ：王金鹏
@Date    ：2025/4/8
"""

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from train2 import DQNAgent, CryptoTradingEnv
# ✅ 设置中文字体
plt.rcParams['font.family'] = 'SimHei'
plt.rcParams['axes.unicode_minus'] = False
# 加载训练用的数据（用于归一化）
train_df = pd.read_csv("1hcrypto_data20250402153513.csv")
train_env = CryptoTradingEnv(train_df)

# 加载回测测试数据（2025年4月2日至4月8日）
test_df = pd.read_csv("1hcrypto_data20250408092012.csv")
test_env = CryptoTradingEnv(test_df)

# 统一归一化参数（使用训练集的 min/max）
test_env.min_values = train_env.min_values
test_env.max_values = train_env.max_values
test_env._normalize_data()

# 加载训练好的模型
agent = DQNAgent(state_dim=9, action_dim=3)
agent.model.load_state_dict(torch.load("models/dqn_20250402_154113_episodes_1000_epsilon_0.01.pth"))
agent.model.eval()
agent.epsilon = 0.0  # 禁用探索，纯策略评估

# 初始化环境
state = test_env.reset()
done = False
portfolio_values = []
actions = []
timestamps = []

# 回测执行
while not done:
    action = agent.select_action(state)
    next_state, reward, done, info = test_env.step(action)
    state = next_state

    # 记录资产和动作
    current_value = test_env.balance + test_env.crypto_held * test_env.denormalize_price(
        test_env.df.loc[test_env.current_step, "close"]
    )
    portfolio_values.append(current_value)
    actions.append(action)
    timestamps.append(test_env.df.loc[test_env.current_step, "timestamp"] if "timestamp" in test_env.df.columns else test_env.current_step)

# 最终收益统计
total_return = portfolio_values[-1] - test_env.initial_balance
print(f"✅ 回测完成，初始资金: {test_env.initial_balance:.2f} USDT")
print(f"📈 期末资产: {portfolio_values[-1]:.2f} USDT")
print(f"💰 总收益: {total_return:.2f} USDT")

# 可视化资产变化曲线
plt.figure(figsize=(10, 5))
plt.plot(portfolio_values, label="资产总值")
plt.title("DQN 回测期间账户价值")
plt.xlabel("时间步")
plt.ylabel("资产价值（USDT）")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
