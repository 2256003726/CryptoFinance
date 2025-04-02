import os
from datetime import datetime

import matplotlib
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
from fetch_data import fetch_data, preprocess_data
from model import TransformerPredictor


def create_sequences(data, seq_length=48):
    X, y = [], []
    for i in range(len(data) - seq_length - 1):
        X.append(data[i:i + seq_length])
        y.append(data[i + seq_length, 3])
    return np.array(X), np.array(y)


# 计算 RMSE（均方根误差）
def rmse(y_true, y_pred):
    return torch.sqrt(torch.mean((y_true - y_pred) ** 2))


# 计算 SMAPE（对称平均绝对百分比误差）
def smape(y_true, y_pred):
    diff = torch.abs(y_true - y_pred)
    sum_ = torch.abs(y_true) + torch.abs(y_pred)
    return 2 * torch.mean(diff / (sum_ + 1e-8)) * 100  # 避免除零错误





if __name__ == '__main__':
    # 用于记录训练过程中的损失、RMSE 和 SMAPE
    losses = []
    rmse_values = []
    smape_values = []
    # 检查是否有可用的 GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 输出当前设备
    print("当前使用的设备:", device)
    # 初始化模型
    model = TransformerPredictor(input_dim=5)
    # 将模型移到设备上（如果有 GPU，就使用 GPU）
    model.to(device)
    # 获取数据
    df = fetch_data()
    df_scaled, scaler = preprocess_data(df)
    X, y = create_sequences(df_scaled, seq_length=48)
    # 将数据移到设备上
    X_tensor = torch.tensor(X, dtype=torch.float32).to(device)  # 移动数据到设备
    y_tensor = torch.tensor(y, dtype=torch.float32).to(device)  # 移动标签到设备
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    epochs = 100
    # 模型的参数
    d_model = 64
    nhead = 4
    num_layers = 3
    batch_size = 32
    for epoch in range(epochs):
        epoch_losses = []
        y_true_all, y_pred_all = [], []

        for i in range(0, len(X_tensor), batch_size):
            X_batch = X_tensor[i:i + batch_size]
            y_batch = y_tensor[i:i + batch_size]
            optimizer.zero_grad()
            output = model(X_batch).squeeze()

            loss = criterion(output, y_batch)
            loss.backward()
            optimizer.step()

            epoch_losses.append(loss.item())
            y_true_all.append(y_batch)
            y_pred_all.append(output)

        # 计算 RMSE 和 MAPE
        y_true_all = torch.cat(y_true_all)
        y_pred_all = torch.cat(y_pred_all)
        train_rmse = rmse(y_true_all, y_pred_all).item()
        train_smape = smape(y_true_all, y_pred_all).item()

        # 记录每个 epoch 的结果
        losses.append(np.mean(epoch_losses))
        rmse_values.append(train_rmse)
        smape_values.append(train_smape)

        if epoch % 5 == 0:
            print(
                f"Epoch {epoch}, Loss: {np.mean(epoch_losses):.4f}, RMSE: {train_rmse:.4f}, SMAPE: {train_smape:.2f}%")
    # 训练完成后，绘制损失、RMSE 和 SMAPE 图形
    epochs_range = range(epochs)
    matplotlib.use('TkAgg')  # 或者 'Agg'
    # 创建图形
    fig, ax = plt.subplots(3, 1, figsize=(10, 12))

    # 绘制 Loss 曲线
    ax[0].plot(epochs_range, losses, label='Loss', color='blue')
    ax[0].set_title('Loss over Epochs')
    ax[0].set_xlabel('Epoch')
    ax[0].set_ylabel('Loss')
    ax[0].legend()

    # 绘制 RMSE 曲线
    ax[1].plot(epochs_range, rmse_values, label='RMSE', color='green')
    ax[1].set_title('RMSE over Epochs')
    ax[1].set_xlabel('Epoch')
    ax[1].set_ylabel('RMSE')
    ax[1].legend()

    # 绘制 SMAPE 曲线
    ax[2].plot(epochs_range, smape_values, label='SMAPE', color='red')
    ax[2].set_title('SMAPE over Epochs')
    ax[2].set_xlabel('Epoch')
    ax[2].set_ylabel('SMAPE (%)')
    ax[2].legend()

    # 显示图形
    plt.tight_layout()
    plt.show()
    # 预测
    model.eval()
    with torch.no_grad():
        # 获取最后一个样本的预测结果
        test_pred = model(X_tensor[-1].unsqueeze(0))

        # 反归一化时，手动填充其他列（假设用当前的 close 价填充其他特征）
        predicted_close = test_pred.item()
        last_close = df_scaled[-1, 3]  # 取归一化后的最后一个 close 价格
        filled_input = np.array(
            [[last_close, last_close, last_close, predicted_close, last_close]])  # 填充 open, high, low, volume 列

        # 使用归一化器对预测结果进行反归一化
        test_pred = scaler.inverse_transform(filled_input)  # 反归一化

        print("预测的 BNB 价格：", test_pred[0][3])  # 输出反归一化后的 close 价格

    import time

    # 创建 models 目录（如果不存在）
    os.makedirs('models', exist_ok=True)

    # 获取当前时间并格式化为 YYYYMMDDHHMM 格式
    current_time = datetime.now().strftime("%Y%m%d%H%M")

    # 动态生成文件名，包含模型的参数信息
    filename = f"models/transformer_model_d{d_model}_n{nhead}_l{num_layers}_b{batch_size}_{current_time}.pth"

    # 保存模型
    torch.save(model.state_dict(), filename)
    print(f"模型已保存为 {filename}！")
