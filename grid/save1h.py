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
output_file = "trade_suggestions_1h.csv"


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

    # 根据 RSI 判断（动态评分）
    if last_rsi < 30:
        rsi_score = max(1, 2 - last_rsi / 15)  # RSI 越低，得分越高
        score += rsi_score
        reasons.append(f"RSI 低于 30（{last_rsi:.2f}），市场可能超卖（+{rsi_score:.2f}）")
    elif last_rsi > 70:
        rsi_score = max(1, (last_rsi - 70) / 15 + 1)  # RSI 越高，扣分越高
        score -= rsi_score
        reasons.append(f"RSI 高于 70（{last_rsi:.2f}），市场可能超买（-{rsi_score:.2f}）")

    # 根据 MACD 判断（动态评分）
    macd_diff = last_macd - last_macd_signal
    if macd_diff > 0:
        macd_score = min(2, macd_diff * 2)  # 差距越大，分数越高
        score += macd_score
        reasons.append(f"MACD 高于信号线，市场上升趋势（+{macd_score:.2f}）")
    elif macd_diff < 0:
        macd_score = min(2, abs(macd_diff) * 2)
        score -= macd_score
        reasons.append(f"MACD 低于信号线，市场可能下跌（-{macd_score:.2f}）")

    # 根据 SMA 判断（动态评分）
    sma_diff = last_sma_50 - last_sma_100
    if sma_diff > 0:
        sma_score = min(2.5, max(0.5, sma_diff / last_sma_100 * 50))  # 50均线上穿100均线的幅度决定得分
        score += sma_score
        reasons.append(f"50均线上穿100均线（+{sma_score:.2f}）")
    elif sma_diff < 0:
        sma_score = min(2.5, max(0.5, abs(sma_diff) / last_sma_100 * 50))
        score -= sma_score
        reasons.append(f"50均线下穿100均线（-{sma_score:.2f}）")

    # 根据布林带判断（动态评分）
    bb_upper_diff = last_close - last_bb_upper
    bb_lower_diff = last_close - last_bb_lower
    if bb_upper_diff >= 0:
        bb_score = min(2.0, max(0.5, bb_upper_diff / last_bb_upper * 50))  # 远离上轨，卖出信号更强
        score -= bb_score
        reasons.append(f"价格触及布林带上轨，市场可能超买（-{bb_score:.2f}）")
    elif bb_lower_diff <= 0:
        bb_score = min(2.0, max(0.5, abs(bb_lower_diff) / last_bb_lower * 10))  # 远低于下轨，买入信号更强
        score += bb_score
        reasons.append(f"价格触及布林带下轨，市场可能超卖（+{bb_score:.2f}）")

    # 综合评分判断交易建议
    if score >= 1.5:
        suggestion = "✅ 可能买入机会"
    elif score <= -1.5:
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
        live_trading_suggestions(timeframe="1h")  # 每次获取1小时K线
        time.sleep(3600)  # 每小时更新一次
