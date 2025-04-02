# import numpy as np
# import torch
#
# from deeplearning.train import create_sequences
# from model import TransformerPredictor  # 或者其他你使用的模型
# from fetch_data import fetch_data, preprocess_data
#
#
# # 载入模型函数
# def load_model(model_path):
#     model = TransformerPredictor(input_dim=5)  # 根据你的模型类型调整输入维度
#     model.load_state_dict(torch.load(model_path))
#     model.eval()  # 设置模型为评估模式
#     return model
#
#
# # 读取最新数据并预处理
# def prepare_input_data():
#     df = fetch_data()  # 获取最新数据
#     df_scaled, scaler = preprocess_data(df)  # 预处理数据（归一化等）
#
#     # 假设你使用过去 48 小时的数据作为输入
#     # 使用滑动窗口生成数据
#     X, _ = create_sequences(df_scaled, seq_length=48)
#
#     # 返回最新的输入数据
#     return torch.tensor(X[-1:], dtype=torch.float32), scaler  # 取最后一个样本作为输入
#
#
# # 预测函数
# # 预测函数
# def predict_future(model, input_tensor, scaler, df_scaled):
#     with torch.no_grad():
#         # 使用模型预测
#         prediction = model(input_tensor).squeeze().numpy()  # 去除多余的维度
#
#         # 获取最后一个样本的归一化特征
#         last_close = df_scaled[-1, 3]  # 获取归一化后的最后一个 close 价格
#         last_open = df_scaled[-1, 0]  # 获取归一化后的最后一个 open 价格
#         last_high = df_scaled[-1, 1]  # 获取归一化后的最后一个 high 价格
#         last_low = df_scaled[-1, 2]  # 获取归一化后的最后一个 low 价格
#         last_volume = df_scaled[-1, 4]  # 获取归一化后的最后一个 volume 价格
#
#         # 将预测的 close 填充到其他列
#         filled_input = np.array(
#             [[last_open, last_high, last_low, prediction, last_volume]]
#         )  # 填充其他特征列，假设用预测的 close 值填充
#
#         # 使用归一化器对预测结果进行反归一化
#         prediction = scaler.inverse_transform(filled_input)  # 反归一化
#
#         return prediction[0][3]  # 返回反归一化后的 close 价格
#
#
# # 主程序
# if __name__ == "__main__":
#     # 设置模型路径
#     model_path = 'models/transformer_model_d64_n4_l3_b32_202503271527.pth'  # 替换为实际路径
#
#     # 加载训练好的模型
#     model = load_model(model_path)
#
#     # 准备最新的输入数据
#     input_tensor, scaler = prepare_input_data()
#
#     # 获取最新的归一化数据
#     df = fetch_data()  # 获取最新的数据
#     df_scaled, _ = preprocess_data(df)  # 预处理数据
#
#     # 预测未来的价格
#     predicted_price = predict_future(model, input_tensor, scaler, df_scaled)
#
#     # 输出预测结果
#     print("预测的未来价格：", predicted_price)
import time
import torch
from deeplearning.train import create_sequences
from model import TransformerPredictor
from fetch_data import fetch_data, preprocess_data

# 载入模型函数
def load_model(model_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TransformerPredictor(input_dim=5).to(device)  # 适配 GPU
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model, device

# 读取最新数据并预处理
def prepare_input_data():
    df = fetch_data()
    df_scaled, scaler = preprocess_data(df)

    X, _ = create_sequences(df_scaled, seq_length=48)
    return torch.tensor(X[-1:], dtype=torch.float32), scaler

# 预测函数
def predict_future(model, input_tensor, scaler, device):
    input_tensor = input_tensor.to(device)  # 适配 GPU/CPU
    with torch.no_grad():
        prediction = model(input_tensor).squeeze().cpu().numpy()  # 转回 CPU

        # **正确的反归一化方式**
        prediction = scaler.inverse_transform([[0, 0, 0, prediction, 0]])[0][3]

    return prediction

    # def predict_future(model, input_tensor, scaler, device):
#     input_tensor = input_tensor.to(device)  # 适配 GPU/CPU
#     with torch.no_grad():
#         prediction = model(input_tensor).squeeze().cpu().numpy()  # 转回 CPU
#
#         # 反归一化
#         last_close = scaler.data_min_[3]  # 取归一化前的 close 价格
#         filled_input = [[last_close, last_close, last_close, prediction, last_close]]
#         prediction = scaler.inverse_transform(filled_input)[0][3]

    return prediction
if __name__ == "__main__":
    model_path = 'models/transformer_model_d64_n4_l3_b32_202503271617.pth'  # 替换为你的模型路径
    model, device = load_model(model_path)

    last_prediction = None  # 存储上一分钟的预测值

    while True:
        try:
            # 获取当前市场数据并进行预测
            input_tensor, scaler = prepare_input_data()
            predicted_price = predict_future(model, input_tensor, scaler, device)

            # 记录当前时间
            current_time = time.strftime('%Y-%m-%d %H:%M:%S')
            print(f"[{current_time}] 预测的未来价格：{predicted_price:.2f}")

            # **检查上一分钟预测值与实际价格**
            if last_prediction is not None:
                time.sleep(60)  # 等待 1 分钟获取实际价格

                # 获取最新的 1m K 线数据
                df = fetch_data()
                actual_price = df.iloc[-1]["close"]  # 获取最新的 close 价格

                # 计算误差
                error = abs(predicted_price - actual_price)
                percentage_error = (error / actual_price) * 100

                print(f"[{current_time}] 实际价格：{actual_price:.2f}, 预测误差：{error:.2f}（{percentage_error:.2f}%）")

            # 更新上一分钟预测值
            last_prediction = predicted_price

        except Exception as e:
            print("预测出错:", e)
            time.sleep(10)  # 出错后等待 10 秒继续尝试
# **实时预测主程序**
# if __name__ == "__main__":
#     model_path = 'models/transformer_model_d64_n4_l3_b32_202503271617.pth'  # 替换为你的模型路径
#     model, device = load_model(model_path)
#
#     while True:
#         try:
#             input_tensor, scaler = prepare_input_data()
#             predicted_price = predict_future(model, input_tensor, scaler, device)
#             print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] 预测的未来价格：{predicted_price:.2f}")
#
#             time.sleep(60)  # 每 60 秒更新一次预测（可调整）
#         except Exception as e:
#             print("预测出错:", e)
#             time.sleep(10)  # 出错后等待 10 秒继续尝试
