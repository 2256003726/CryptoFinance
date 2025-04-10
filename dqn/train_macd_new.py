#!/usr/bin/env python
# -*- coding: UTF-8 -*-
import numpy as np
import pandas as pd
import gym
from gym import spaces
import torch
import random
from collections import deque


class CryptoConfig:
    """配置参数"""

    def __init__(self):
        # 数据参数
        self.window_size = 30
        self.feature_min = {'close': 30000}  # 需根据实际数据计算
        self.feature_max = {'close': 60000}

        # 交易参数
        self.initial_balance = 100000
        self.max_position = 10
        self.fee_rate = 0.001  # 0.1%手续费
        self.margin_ratio = 0.5  # 空头保证金比例
        self.stop_loss_ratio = 0.5  # 亏损50%强平

        # 训练参数
        self.batch_size = 64
        self.memory_capacity = 10000


class CryptoEnv(gym.Env):
    """加密货币多空交易环境"""

    def __init__(self, df, config):
        super().__init__()
        self.df = df
        self.config = config

        # 动作空间：0=持仓，1=做多，2=做空
        self.action_space = spaces.Discrete(3)

        # 状态空间：[市场数据, 净仓位, 余额, 盈亏比, 波动率]
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(config.window_size, len(df.columns) + 4),
            dtype=np.float32
        )

        self.reset()

    def reset(self):
        """重置环境状态"""
        self.current_step = self.config.window_size
        self.balance = self.config.initial_balance
        self.net_position = 0  # 净仓位（正=多仓，负=空仓）
        self.avg_entry_price = 0
        self.portfolio_history = [self.config.initial_balance]
        return self._get_state()

    def step(self, action):
        """执行交易动作"""
        # 获取当前价格
        current_price = self._get_current_price()

        # 初始化变量
        reward = 0
        done = False
        info = {"action": action}

        # === 执行交易 ===
        if action == 1:  # 做多
            self._execute_long(current_price)
        elif action == 2:  # 做空
            self._execute_short(current_price)

        # === 计算资产状态 ===
        total_value = self._calculate_total_assets(current_price)
        unrealized_pnl = self._calculate_unrealized_pnl(current_price)

        # === 奖励函数 ===
        reward = self._calculate_reward(current_price, unrealized_pnl)

        # === 更新状态 ===
        self.current_step += 1
        self.portfolio_history.append(total_value)

        # === 终止条件 ===
        if (total_value <= self.config.initial_balance * self.config.stop_loss_ratio or
                self.current_step >= len(self.df) - 1):
            done = True

        return self._get_state(), reward, done, info

    # ========== 核心交易方法 ==========
    def _execute_long(self, price):
        """执行做多操作"""
        if self.balance >= price * (1 + self.config.fee_rate):
            cost = price * (1 + self.config.fee_rate)
            self.balance -= cost
            self.net_position += 1
            self._update_entry_price(price)

    def _execute_short(self, price):
        """执行做空操作"""
        margin_required = price * self.config.margin_ratio
        if self.balance >= margin_required:
            self.balance -= margin_required
            self.net_position -= 1
            self._update_entry_price(price)
            self.balance += price * (1 - self.config.fee_rate)

    def _update_entry_price(self, price):
        """更新平均开仓价"""
        prev_position = self.net_position - 1 if self.net_position > 0 else self.net_position + 1
        total_cost = abs(prev_position) * self.avg_entry_price + price
        self.avg_entry_price = total_cost / max(1, abs(self.net_position))

    # ========== 状态计算方法 ==========
    def _get_current_price(self):
        """获取当前价格（反归一化）"""
        price_norm = self.df.loc[self.current_step, "close"]
        return (price_norm * (self.config.feature_max['close'] - self.config.feature_min['close'])
                + self.config.feature_min['close'])

    def _calculate_total_assets(self, price):
        """计算总资产"""
        return self.balance + self.net_position * price

    def _calculate_unrealized_pnl(self, price):
        """计算未实现盈亏"""
        if self.net_position == 0:
            return 0
        return (price - self.avg_entry_price) * self.net_position

    def _get_state(self):
        """构建观察状态"""
        state = np.zeros((self.config.window_size, len(self.df.columns) + 4))

        # 市场数据窗口
        state[:, :-4] = self.df.iloc[
                        self.current_step - self.config.window_size: self.current_step
                        ].values

        # 账户状态特征
        current_price = self._get_current_price()
        state[:, -4] = self.net_position / self.config.max_position  # 仓位比例
        state[:, -3] = self.balance / self.config.initial_balance  # 资金利用率
        state[:, -2] = self._calculate_unrealized_pnl(current_price) / self.config.initial_balance  # 盈亏比
        state[:, -1] = self.df['volatility'].iloc[self.current_step]  # 波动率

        return state

    # ========== 奖励函数 ==========
    def _calculate_reward(self, price, unrealized_pnl):
        """多因子奖励函数"""
        # 基础盈亏奖励
        reward = unrealized_pnl * 0.01

        # 仓位风险惩罚
        position_penalty = abs(self.net_position) * 0.001
        reward -= position_penalty

        # 趋势跟随奖励
        price_diff = price - self._get_previous_price()
        if (self.net_position > 0 and price_diff > 0) or (self.net_position < 0 and price_diff < 0):
            reward += abs(price_diff) * 0.02

        return reward

    def _get_previous_price(self):
        """获取上一步价格"""
        prev_price_norm = self.df.loc[self.current_step - 1, "close"]
        return (prev_price_norm * (self.config.feature_max['close'] - self.config.feature_min['close'])
                + self.config.feature_min['close'])


class CryptoAgent:
    """交易智能体"""

    def __init__(self, env, config):
        self.env = env
        self.config = config
        self.memory = deque(maxlen=config.memory_capacity)
        self.epsilon = 1.0

        # 初始化Q网络
        self.q_net = self._build_network()
        self.target_net = self._build_network()
        self.update_target_network()

    def _build_network(self):
        """构建神经网络"""
        # 示例网络结构，实际应根据状态维度调整
        return torch.nn.Sequential(
            torch.nn.Linear(np.prod(self.env.observation_space.shape), 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, self.env.action_space.n)
        )

    def select_action(self, state):
        """选择交易动作"""
        if random.random() < self.epsilon:
            return random.choice([0, 1, 2])

        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            q_values = self.q_net(state_tensor)
        return torch.argmax(q_values).item()

    def store_experience(self, state, action, reward, next_state, done):
        """存储交易经验"""
        self.memory.append((state, action, reward, next_state, done))

    def train(self):
        """训练网络"""
        if len(self.memory) < self.config.batch_size:
            return

        batch = random.sample(self.memory, self.config.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        # 转换为张量
        states = torch.FloatTensor(np.array(states))
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(np.array(next_states))
        dones = torch.FloatTensor(dones)

        # 计算Q值更新
        current_q = self.q_net(states).gather(1, actions.unsqueeze(1))
        next_q = self.target_net(next_states).max(1)[0]
        target_q = rewards + (1 - dones) * 0.99 * next_q

        # 计算损失
        loss = torch.nn.MSELoss()(current_q.squeeze(), target_q)

        # 反向传播
        optimizer = torch.optim.Adam(self.q_net.parameters())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        return loss.item()

    def update_target_network(self):
        """更新目标网络"""
        self.target_net.load_state_dict(self.q_net.state_dict())

    def update_epsilon(self):
        """衰减探索率"""
        self.epsilon = max(0.01, self.epsilon * 0.995)


def main():
    """主训练流程"""
    # 示例数据加载（需替换为实际数据）
    df = pd.DataFrame({
        'close': np.random.normal(45000, 5000, 1000),
        'volatility': np.random.uniform(0.01, 0.05, 1000)
    })

    config = CryptoConfig()
    env = CryptoEnv(df, config)
    agent = CryptoAgent(env, config)

    # 训练循环
    for episode in range(100):
        state = env.reset()
        total_reward = 0

        while True:
            action = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)

            agent.store_experience(state, action, reward, next_state, done)
            loss = agent.train()

            total_reward += reward
            state = next_state

            if done:
                break

        agent.update_epsilon()
        if episode % 10 == 0:
            agent.update_target_network()
            print(f"Episode {episode}, Total Reward: {total_reward:.2f}, Epsilon: {agent.epsilon:.2f}")


if __name__ == "__main__":
    main()