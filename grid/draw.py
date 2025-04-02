import pandas as pd
import matplotlib.pyplot as plt

import matplotlib
matplotlib.use('TkAgg')  # 可选 'Agg', 'Qt5Agg' 等
def plot_price_and_score(csv_file):
    """
    读取 CSV 文件，并绘制价格 (last_close) 和 score 的趋势图。
    """
    # 读取 CSV 数据
    df = pd.read_csv(csv_file, parse_dates=['timestamp'])
    df = df.sort_values('timestamp')  # 确保按时间排序

    # 创建图表
    fig, ax1 = plt.subplots(figsize=(12, 6))

    # 绘制价格曲线
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Price', color='tab:blue')
    ax1.plot(df['timestamp'], df['last_close'], color='tab:blue', label='Price')
    ax1.tick_params(axis='y', labelcolor='tab:blue')

    # 创建第二个 Y 轴（共享 X 轴）
    ax2 = ax1.twinx()
    ax2.set_ylabel('Score', color='tab:red')
    ax2.plot(df['timestamp'], df['score'], color='tab:red', linestyle='dashed', label='Score')
    ax2.tick_params(axis='y', labelcolor='tab:red')

    # 标题 & 图例
    fig.suptitle('Price and Score Trend')
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')

    # 旋转 X 轴标签，防止重叠
    plt.xticks(rotation=45)
    plt.grid()
    plt.show()


plot_price_and_score('trade_suggestions_1h.csv')
