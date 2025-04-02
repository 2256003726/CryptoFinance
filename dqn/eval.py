import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from train2 import DQN, CryptoTradingEnv

# 读取市场数据
df = pd.read_csv("crypto_data.csv")
env = CryptoTradingEnv(df)

# 加载训练好的 DQN 模型
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n
dqn = DQN(state_dim, action_dim)
dqn.load_state_dict(torch.load("models/dqn_20250328_153409_episodes_1000_epsilon_0.01.pth"))
dqn.eval()

state = env.reset()
total_rewards = []

while True:
    state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
    action = torch.argmax(dqn(state_tensor)).item()
    next_state, reward, done, _ = env.step(action)
    total_rewards.append(reward)
    state = next_state
    if done:
        break

# 绘制收益曲线
plt.plot(np.cumsum(total_rewards), label='Cumulative Reward')
plt.xlabel('Time Steps')
plt.ylabel('Total Reward')
plt.title('DQN Trading Performance')
plt.legend()
plt.show()
