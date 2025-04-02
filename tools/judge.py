import pandas as pd
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('TkAgg')  # 可选 'Agg', 'Qt5Agg' 等
# 设置中文字体（确保系统中安装了支持中文的字体，如SimHei等）
matplotlib.rcParams['font.family'] = 'SimHei'  # 设置为黑体
matplotlib.rcParams['axes.unicode_minus'] = False  # 解决负号 '-' 显示问题
# 读取 CSV 文件
df = pd.read_csv("../grid/trade_suggestions_1.csv", parse_dates=["timestamp"])
df.sort_values(by="timestamp", inplace=True)


# 基准回测函数
def baseline_backtest(df, initial_balance=10000):
    # 基准持仓策略（不做任何买卖，仅持有）
    df["baseline_net_value"] = df["last_close"] / df["last_close"].iloc[0] * initial_balance
    return df


# 策略回测函数
def backtest(df, initial_balance=10000, trade_size=1000):
    balance = initial_balance  # 初始账户余额
    position = 0  # 持仓数量
    trade_log = []  # 交易日志
    net_value = []  # 策略净值

    for index, row in df.iterrows():
        price = row["last_close"]
        score = row["score"]
        suggestion = row["suggestion"]

        if score >= 2 and balance >= trade_size:
            # 以 trade_size 金额买入
            buy_amount = trade_size / price
            position += buy_amount
            balance -= trade_size
            trade_log.append((row["timestamp"], "BUY", price, trade_size))

        elif score <= -2 and position > 0:
            # 卖出所有持仓
            sell_value = position * price
            balance += sell_value
            position = 0
            trade_log.append((row["timestamp"], "SELL", price, sell_value))

        # 计算当前净值
        current_value = balance + (position * price)
        net_value.append(current_value)

    # 添加策略净值到DataFrame
    df["strategy_net_value"] = pd.Series(net_value, index=df.index)

    # 计算最终净值
    final_value = balance + (position * df.iloc[-1]["last_close"])
    return trade_log, df, final_value


# 执行回测
df = baseline_backtest(df)  # 基准回测
trade_log2, df_with_strategy, final_value = backtest(df, initial_balance=10000)

# 输出交易日志
for trade in trade_log2:
    print(trade)

# 绘制收益曲线
plt.figure(figsize=(10, 5))
plt.plot(df_with_strategy["timestamp"], df_with_strategy["baseline_net_value"], label="基准净值", linestyle="--")
plt.plot(df_with_strategy["timestamp"], df_with_strategy["strategy_net_value"], label="策略净值")
plt.xlabel("日期")
plt.ylabel("净值")
plt.legend()
plt.title("策略与基准净值对比")
plt.show()

print(f"最终投资组合净值（策略）：{final_value}")
