import gym
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import random
import collections
from gym import spaces


class CryptoTradingEnv(gym.Env):
    def __init__(self, df, initial_balance=10000):
        super(CryptoTradingEnv, self).__init__()
        self.df = df.reset_index()
        self.initial_balance = initial_balance
        self.current_step = 0
        self.balance = initial_balance
        self.crypto_held = 0
        self.total_reward = 0
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(low=0, high=np.inf, shape=(5,), dtype=np.float32)

    def reset(self):
        self.current_step = 0
        self.balance = self.initial_balance
        self.crypto_held = 0
        self.total_reward = 0
        return self._next_observation()

    def _next_observation(self):
        obs = np.array([
            self.df.loc[self.current_step, "open"],
            self.df.loc[self.current_step, "high"],
            self.df.loc[self.current_step, "low"],
            self.df.loc[self.current_step, "close"],
            self.crypto_held
        ])
        return obs

    def step(self, action):
        current_price = self.df.loc[self.current_step, "close"]
        reward = 0
        if action == 1 and self.balance >= current_price:
            self.crypto_held += 1
            self.balance -= current_price
        elif action == 2 and self.crypto_held > 0:
            self.crypto_held -= 1
            self.balance += current_price
            reward += current_price - self.df.loc[self.current_step - 1, "close"]
        self.current_step += 1
        done = self.current_step >= len(self.df) - 1
        self.total_reward += reward
        return self._next_observation(), reward, done, {}


class DQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


class DQNAgent:
    def __init__(self, state_dim, action_dim):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = DQN(state_dim, action_dim).to(self.device)
        self.target_model = DQN(state_dim, action_dim).to(self.device)
        self.target_model.load_state_dict(self.model.state_dict())
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.loss_fn = nn.MSELoss()
        self.memory = collections.deque(maxlen=10000)
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01

    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, 2)
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            return torch.argmax(self.model(state)).item()

    def store_experience(self, experience):
        self.memory.append(experience)

    def train(self, batch_size=64):
        if len(self.memory) < batch_size:
            return
        batch = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        states = torch.tensor(states, dtype=torch.float32).to(self.device)
        actions = torch.tensor(actions, dtype=torch.long).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        next_states = torch.tensor(next_states, dtype=torch.float32).to(self.device)
        dones = torch.tensor(dones, dtype=torch.float32).to(self.device)
        q_values = self.model(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        next_q_values = self.target_model(next_states).max(1)[0]
        target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
        loss = self.loss_fn(q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        if random.random() < 0.1:
            self.target_model.load_state_dict(self.model.state_dict())

    def update_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)


def train_dqn():
    df = pd.read_csv("crypto_data.csv")
    env = CryptoTradingEnv(df)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    agent = DQNAgent(state_dim, action_dim)
    num_episodes = 100
    for episode in range(num_episodes):
        state = env.reset()
        total_reward = 0
        for t in range(200):
            action = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)
            agent.store_experience((state, action, reward, next_state, done))
            agent.train()
            state = next_state
            total_reward += reward
            if done:
                break
        agent.update_epsilon()
        print(f"Episode {episode + 1}: Total Reward = {total_reward}")
    print("训练完成！")


if __name__ == "__main__":
    train_dqn()