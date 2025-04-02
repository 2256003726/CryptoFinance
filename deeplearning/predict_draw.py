import time

import matplotlib
import pandas as pd
import matplotlib.pyplot as plt
from collections import deque

from deeplearning.fetch_data import fetch_data
from deeplearning.predict import prepare_input_data, predict_future, load_model
matplotlib.use('TkAgg')  # 可选 'Agg', 'Qt5Agg' 等

plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体显示中文
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
# 存储历史数据
timestamps = deque(maxlen=100)  # 存储最近 100 次预测的时间
predictions = deque(maxlen=100)  # 存储最近 100 次的预测值
actual_prices = deque(maxlen=100)  # 存储最近 100 次的真实价格
errors = deque(maxlen=100)  # 存储最近 100 次的误差

# 初始化 Matplotlib
plt.ion()  # 允许动态图
fig, ax = plt.subplots(figsize=(10, 5))

# CSV 文件路径
csv_file = "predictions_log.csv"


def save_to_csv(timestamp, predicted, actual, error):
    """保存数据到 CSV 文件"""
    df = pd.DataFrame([[timestamp, predicted, actual, error]],
                      columns=["Time", "Predicted", "Actual", "Error"])

    # 追加模式写入
    df.to_csv(csv_file, mode='a', index=False, header=not pd.io.common.file_exists(csv_file))

import matplotlib.dates as mdates
def update_plot():
    """更新实时预测误差图"""
    ax.clear()
    ax.plot(timestamps, predictions, label="预测价格", linestyle="--", marker="o")
    ax.plot(timestamps, actual_prices, label="真实价格", linestyle="-", marker="x")
    ax.fill_between(timestamps, predictions, actual_prices, color='gray', alpha=0.2)  # 误差阴影
    # 设置 x 轴时间格式为 "YYYY-MM-DD HH:MM"
    # ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d %H:%M"))
    ax.set_xlabel("时间")
    ax.set_ylabel("价格")
    ax.set_title("预测 vs 真实价格")
    ax.legend()
    plt.xticks(rotation=45)
    plt.draw()
    plt.pause(0.1)  # 暂停 0.1s 让图像更新
# def update_plot():
#     """更新实时预测误差图"""
#     ax.clear()
#
#     # 绘制预测价格和真实价格
#     ax.plot(timestamps, predictions, label="预测价格", linestyle="--", marker="o", color='blue')
#     ax.plot(timestamps, actual_prices, label="真实价格", linestyle="-", marker="x", color='green')
#     ax.fill_between(timestamps, predictions, actual_prices, color='gray', alpha=0.2)  # 误差阴影
#
#     # 计算误差率（带正负）
#     error_rates = [(p - a) / a * 100 for p, a in zip(predictions, actual_prices)]
#
#     # 创建第二个 y 轴
#     ax2 = ax.twinx()
#
#     # 根据误差的正负，分别绘制不同颜色的误差率
#     ax2.bar(timestamps, error_rates, color=['red' if e > 0 else 'green' for e in error_rates], alpha=0.6,
#             label="误差率 (%)")
#
#     # 设置标签和标题
#     ax.set_xlabel("时间")
#     ax.set_ylabel("价格")
#     ax2.set_ylabel("误差率 (%)")
#     ax.set_title("预测 vs 真实价格 & 误差率（正负方向）")
#
#     # 添加图例
#     ax.legend(loc="upper left")
#     ax2.legend(loc="upper right")
#
#     # 旋转 x 轴刻度
#     plt.xticks(rotation=45)
#
#     # 更新图像
#     plt.draw()
#     plt.pause(0.1)  # 暂停 0.1s 让图像更新


if __name__ == "__main__":
    model_path = 'models/transformer_model_d64_n4_l3_b32_202503272139.pth'  # 替换为你的模型路径
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
                time.sleep(60*60)  # 等待 1 分钟获取实际价格

                # 获取最新的 1m K 线数据
                df = fetch_data()
                actual_price = df.iloc[-1]["close"]  # 获取最新的 close 价格

                # 计算误差
                error = abs(predicted_price - actual_price)
                percentage_error = (error / actual_price) * 100

                print(f"[{current_time}] 实际价格：{actual_price:.2f}, 预测误差：{error:.2f}（{percentage_error:.2f}%）")

                # 记录数据
                timestamps.append(current_time)
                predictions.append(predicted_price)
                actual_prices.append(actual_price)
                errors.append(error)

                # 保存到 CSV
                save_to_csv(current_time, predicted_price, actual_price, error)

                # 更新图表
                update_plot()

            # 更新上一分钟预测值
            last_prediction = predicted_price

        except Exception as e:
            print("预测出错:", e)
            time.sleep(10)  # 出错后等待 10 秒继续尝试
