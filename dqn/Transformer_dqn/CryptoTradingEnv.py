#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：CryptoFinance 
@File    ：CryptoTradingEnv.py
@Author  ：王金鹏
@Date    ：2025/4/8 10:03 
"""
import numpy as np
from gym import spaces
from dqn.train2 import CryptoTradingEnv

class SeqLenCryptoTradingEnv(CryptoTradingEnv):
    def __init__(self, df, seq_len=10, initial_balance=10000):
        super(SeqLenCryptoTradingEnv, self).__init__(df, initial_balance)
        self.seq_len = seq_len
        self.observation_space = spaces.Box(low=0, high=np.inf, shape=(seq_len, 9), dtype=np.float32)  # 修改 observation_space

    def _next_observation(self):
        """返回过去 seq_len 个时间步的数据"""
        obs = []
        for step in range(self.current_step - self.seq_len + 1, self.current_step + 1):
            if step >= 0:
                obs.append([
                    self.df.loc[step, "open"],
                    self.df.loc[step, "high"],
                    self.df.loc[step, "low"],
                    self.df.loc[step, "close"],
                    self.df.loc[step, "volume"],
                    self.df.loc[step, "quote_asset_volume"],
                    self.df.loc[step, "num_trades"],
                    self.df.loc[step, "taker_buy_base"],
                    self.df.loc[step, "taker_buy_quote"]
                ])
        # 补充不足 seq_len 的数据
        while len(obs) < self.seq_len:
            obs.insert(0, np.zeros(9))  # 如果历史数据不足，填充零向量

        return np.array(obs)
