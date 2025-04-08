from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from collections import deque
import random
import matplotlib.pyplot as plt
import os
from CryptoTradingEnv import SeqLenCryptoTradingEnv

# ===== 设置 GPU / CPU =====
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ===== Transformer 模型 =====
class TransformerModel(nn.Module):
    def __init__(self, input_dim, d_model=64, nhead=4, num_layers=2):
        super(TransformerModel, self).__init__()
        self.embedding = nn.Linear(input_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(d_model, 128)

    def forward(self, x):
        x = self.embedding(x)
        x = self.transformer_encoder(x)
        return self.fc_out(x[:, -1])  # 用最后一个时间步的输出

# ===== DQN 头部 =====
class DQN(nn.Module):
    def __init__(self, input_dim, action_dim):
        super(DQN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        )

    def forward(self, x):
        return self.net(x)

# ===== 整合 Transformer + DQN 的智能体 =====
class TransformerDQNAgent:
    def __init__(self, state_dim, action_dim, seq_len, lr=1e-4, gamma=0.99, epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995):
        self.seq_len = seq_len
        self.action_dim = action_dim
        self.transformer = TransformerModel(state_dim).to(device)
        self.dqn = DQN(128, action_dim).to(device)
        self.optimizer = optim.Adam(list(self.transformer.parameters()) + list(self.dqn.parameters()), lr=lr)
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.memory = deque(maxlen=10000)
        self.batch_size = 64

    def select_action(self, state_seq):
        if np.random.rand() < self.epsilon:
            return random.randrange(self.action_dim)
        state_tensor = torch.FloatTensor(state_seq).unsqueeze(0).to(device)  # [1, seq_len, state_dim]
        with torch.no_grad():
            features = self.transformer(state_tensor)
            q_values = self.dqn(features)
        return q_values.argmax().item()

    def store_transition(self, state_seq, action, reward, next_state_seq, done):
        self.memory.append((state_seq, action, reward, next_state_seq, done))

    def train_step(self):
        if len(self.memory) < self.batch_size:
            return
        batch = random.sample(self.memory, self.batch_size)
        state_seqs, actions, rewards, next_state_seqs, dones = zip(*batch)

        state_seqs = torch.FloatTensor(np.array(state_seqs)).to(device)
        next_state_seqs = torch.FloatTensor(np.array(next_state_seqs)).to(device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(device)

        features = self.transformer(state_seqs)
        q_values = self.dqn(features).gather(1, actions)

        with torch.no_grad():
            next_features = self.transformer(next_state_seqs)
            next_q_values = self.dqn(next_features).max(1)[0].unsqueeze(1)
            target_q_values = rewards + self.gamma * next_q_values * (1 - dones)

        loss = nn.MSELoss()(q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def save_model(self, path):
        torch.save({
            'transformer_state_dict': self.transformer.state_dict(),
            'dqn_state_dict': self.dqn.state_dict()
        }, path)

# ===== 主程序 =====
if __name__ == "__main__":
    df = pd.read_csv("../1hcrypto_data20250408092012.csv")
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.drop(columns=["timestamp"])

    env = SeqLenCryptoTradingEnv(df, seq_len=10)
    agent = TransformerDQNAgent(state_dim=9, action_dim=3, seq_len=10)

    episodes = 100
    rewards_log = []

    for episode in range(episodes):
        state_seq = env.reset()
        done = False
        total_reward = 0

        while not done:
            action = agent.select_action(state_seq)
            next_state_seq, reward, done, _ = env.step(action)
            agent.store_transition(state_seq, action, reward, next_state_seq, done)
            agent.train_step()
            state_seq = next_state_seq
            total_reward += reward

        rewards_log.append(total_reward)
        print(f"Episode {episode + 1}/{episodes} - Total Reward: {total_reward:.2f}, Epsilon: {agent.epsilon:.4f}")

    # 训练完成后的保存逻辑
    kline_interval = "1h"  # 修改为你实际用的周期
    date_str = datetime.now().strftime("%Y%m%d_%H%M")
    filename = f"transformer_dqn_{kline_interval}_seq{agent.seq_len}_dmodel{agent.transformer.embedding.out_features}_ep{episodes}_{date_str}.pt"

    os.makedirs("checkpoints", exist_ok=True)
    model_path = os.path.join("checkpoints", filename)

    agent.save_model(model_path)

    print(f"✅ 模型已保存：{model_path}")
    # 绘图：训练奖励曲线
    plt.figure(figsize=(10, 5))
    plt.plot(rewards_log, label="Episode Reward")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("Transformer DQN 训练奖励曲线")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("training_reward_curve.png")
    plt.show()
