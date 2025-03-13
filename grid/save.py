import ccxt
import pandas as pd
import talib
import datetime
import time
import os

# 配置交易所（仅查看行情，无需 API Key）
exchange = ccxt.binance({
    'proxies': {
        'http': 'http://127.0.0.1:7890',  # 配置你的代理地址
        'https': 'http://127.0.0.1:7890',  # 配置你的代理地址
    },
    'options': {'defaultType': 'future'}  # 指定合约市场（future）
})


def fetch_ohlcv(symbol="BNB/USDT", timeframe="15m", limit=100):
    """获取市场数据（K线数据）"""
    try:
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        return df
    except Exception as e:
        print(f"[{datetime.datetime.now()}] ❌ 获取行情数据失败: {e}")
        return None


# 文件名固定
output_file = "trade_suggestions_15m.csv"


def save_to_csv(df, suggestion, score, reasons, last_close):
    """将交易建议保存到CSV文件"""
    # 获取简化的时间格式（去掉微秒）
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    # 创建一行新的记录
    new_row = {
        'timestamp': timestamp,
        'last_close': round(last_close, 2),  # 保存当前价格
        'suggestion': suggestion,
        'score': score,
        'RSI': round(df['RSI'].iloc[-1], 2),
        'MACD': round(df['MACD'].iloc[-1], 2),
        'MACD_signal': round(df['MACD_signal'].iloc[-1], 2),
        'SMA_50': round(df['SMA_50'].iloc[-1], 2),
        'SMA_100': round(df['SMA_100'].iloc[-1], 2),
        'BB_upper': round(df['BB_upper'].iloc[-1], 2),  # 布林带上轨
        'BB_lower': round(df['BB_lower'].iloc[-1], 2),  # 布林带下轨
        'reasons': '; '.join(reasons)
    }

    # 将新记录添加到现有的CSV文件中
    try:
        file_exists = os.path.exists(output_file)
        df_new = pd.DataFrame([new_row])

        # 如果文件不存在，添加列名
        df_new.to_csv(output_file, mode='a', header=not file_exists, index=False)

        print(f"[{datetime.datetime.now()}] ✔️ 交易建议已保存到 {output_file}")
    except Exception as e:
        print(f"[{datetime.datetime.now()}] ❌ 保存数据失败: {e}")


def live_trading_suggestions(timeframe="1h"):
    """实时交易建议，根据不同的时间框架进行数据获取和策略判断"""
    print(f"\n[{datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] 📈 获取最新交易建议，时间框架：{timeframe}...")

    df = fetch_ohlcv(timeframe=timeframe)
    if df is None:
        return

    # 计算技术指标（使用 TA-Lib）
    df["RSI"] = talib.RSI(df["close"], timeperiod=14)
    df["MACD"], df["MACD_signal"], _ = talib.MACD(df["close"], fastperiod=12, slowperiod=26, signalperiod=9)
    df["SMA_50"] = talib.SMA(df["close"], timeperiod=50)
    df["SMA_100"] = talib.SMA(df["close"], timeperiod=100)

    # 计算布林带
    df['BB_upper'], df['BB_middle'], df['BB_lower'] = talib.BBANDS(df['close'], timeperiod=20, nbdevup=2, nbdevdn=2,
                                                                   matype=0)

    # 获取最新数据
    # 保留小数点后 2 位
    last_close = df["close"].iloc[-1]
    last_rsi = round(df["RSI"].iloc[-1], 2)
    last_macd = round(df["MACD"].iloc[-1], 2)
    last_macd_signal = round(df["MACD_signal"].iloc[-1], 2)
    last_sma_50 = round(df["SMA_50"].iloc[-1], 2)
    last_sma_100 = round(df["SMA_100"].iloc[-1], 2)
    last_bb_upper = round(df["BB_upper"].iloc[-1], 2)
    last_bb_lower = round(df["BB_lower"].iloc[-1], 2)

    # 评分机制
    score = 0
    reasons = []

    # 根据 RSI 判断
    if last_rsi < 30:
        score += 1
        reasons.append("RSI 低于 30，市场可能超卖（买入信号）")
    elif last_rsi > 70:
        score -= 1
        reasons.append("RSI 高于 70，市场可能超买（卖出信号）")

    # 根据 MACD 判断
    if last_macd > last_macd_signal:
        score += 1.5  # 提高 MACD 的权重
        reasons.append("MACD 高于信号线，市场有上升趋势")
    elif last_macd < last_macd_signal:
        score -= 1.5  # 提高 MACD 的权重
        reasons.append("MACD 低于信号线，市场可能下跌")

    # 根据 SMA 判断
    if last_sma_50 > last_sma_100:
        score += 2
        reasons.append("50均线上穿100均线，长期看涨")
    elif last_sma_50 < last_sma_100:
        score -= 2
        reasons.append("50均线下穿100均线，长期看跌")

    # 布林带判断（增加权重）
    if last_close >= last_bb_upper:
        score -= 1.5  # 提高布林带的卖出信号权重
        reasons.append("价格触及布林带上轨，市场可能超买（卖出信号）")
    elif last_close <= last_bb_lower:
        score += 1.5  # 提高布林带的买入信号权重
        reasons.append("价格触及布林带下轨，市场可能超卖（买入信号）")

    # 综合评分判断交易建议
    if score >= 2:
        suggestion = "✅ 可能买入机会"
    elif score <= -2:
        suggestion = "❌ 可能卖出机会"
    else:
        suggestion = "🔍 持续观察"

    # 打印并保存数据
    print(f"当前价格: {last_close}")
    print(f"交易建议: {suggestion}  综合分: {score}")
    print(
        f"RSI: {last_rsi}, MACD: {last_macd}, MACD_signal: {last_macd_signal}, SMA_50: {last_sma_50}, SMA_100: {last_sma_100}")
    print(f"理由: {', '.join(reasons)}")

    # 保存到CSV文件
    save_to_csv(df, suggestion, score, reasons, last_close)


if __name__ == "__main__":
    while True:
        # live_trading_suggestions(timeframe="1h")  # 每次获取1小时K线
        # time.sleep(3600)  # 每小时更新一次
        live_trading_suggestions(timeframe="15m")  # 每次获取1小时K线
        time.sleep(3600 / 4)  # 每小时更新一次
