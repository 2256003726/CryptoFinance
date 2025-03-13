import ccxt
import pandas as pd
import talib
import datetime
import time

# 配置交易所（仅查看行情，无需 API Key）
exchange = ccxt.binance({
    'proxies': {
        'http': 'http://127.0.0.1:7890',  # 配置你的代理地址
        'https': 'http://127.0.0.1:7890', # 配置你的代理地址
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

def live_trading_suggestions():
    """实时交易建议"""
    print(f"\n[{datetime.datetime.now()}] 📈 获取最新交易建议...")

    df = fetch_ohlcv(symbol="BNB/USDT", timeframe="1h", limit=200)
    if df is None:
        return

    # 计算技术指标
    df["RSI"] = talib.RSI(df["close"], timeperiod=14)
    df["MACD"], df["MACD_signal"], _ = talib.MACD(df["close"], fastperiod=12, slowperiod=26, signalperiod=9)
    df["SMA_50"] = talib.SMA(df["close"], timeperiod=50)
    df["SMA_200"] = talib.SMA(df["close"], timeperiod=200)
    df["BB_middle"], df["BB_upper"], df["BB_lower"] = talib.BBANDS(df["close"], timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)

    # 获取最新数据
    last_close = df["close"].iloc[-1]
    last_rsi = df["RSI"].iloc[-1]
    last_macd = df["MACD"].iloc[-1]
    last_macd_signal = df["MACD_signal"].iloc[-1]
    last_sma_50 = df["SMA_50"].iloc[-1]
    last_sma_200 = df["SMA_200"].iloc[-1]
    last_bb_upper = df["BB_upper"].iloc[-1]
    last_bb_lower = df["BB_lower"].iloc[-1]

    # 评分机制
    score = 0
    reasons = []

    # RSI 信号
    if last_rsi < 30:
        score += 1
        reasons.append("RSI 低于 30，市场可能超卖（买入信号）")
    elif last_rsi > 70:
        score -= 1
        reasons.append("RSI 高于 70，市场可能超买（卖出信号）")

    # MACD 信号
    if last_macd > last_macd_signal:
        score += 1.5
        reasons.append("MACD 高于信号线，市场有上升趋势")
    elif last_macd < last_macd_signal:
        score -= 1.5
        reasons.append("MACD 低于信号线，市场可能下跌")

    # 均线交叉信号
    if last_sma_50 > last_sma_200:
        score += 2
        reasons.append("50均线上穿200均线（黄金交叉），长期看涨")
    elif last_sma_50 < last_sma_200:
        score -= 2
        reasons.append("50均线下穿200均线（死亡交叉），长期看跌")

    # 布林带信号
    if last_close >= last_bb_upper:
        score -= 1.5
        reasons.append("价格触及布林带上轨，市场可能超买（卖出信号）")
    elif last_close <= last_bb_lower:
        score += 1.5
        reasons.append("价格触及布林带下轨，市场可能超卖（买入信号）")

    # 综合评分判断交易建议
    if score >= 3:
        suggestion = "✅ 可能买入机会"
        position_size = 0.5  # 50% 仓位
        leverage = 10  # 使用 10x 杠杆
    elif score >= 2:
        suggestion = "✅ 可能买入机会"
        position_size = 0.3  # 30% 仓位
        leverage = 5  # 使用 5x 杠杆
    elif score >= 0:
        suggestion = "🔍 持续观察"
        position_size = 0.1  # 10% 仓位
        leverage = 1  # 使用 1x 杠杆
    elif score <= -2:
        suggestion = "❌ 可能卖出机会"
        position_size = 0  # 不开仓
        leverage = 1  # 不使用杠杆
    else:
        suggestion = "❌ 可能卖出机会"
        position_size = 0  # 不开仓
        leverage = 1  # 不使用杠杆

    # 输出建议和仓位
    print(f"\n交易建议: {suggestion}  综合分: {score}")
    print(f"仓位建议: {position_size * 100}%")
    print(f"杠杆建议: {leverage}x")
    # 打印理由
    print("评分理由:")
    for reason in reasons:
        print(f"- {reason}")


if __name__ == "__main__":
    while True:
        live_trading_suggestions()
        time.sleep(900)  # 等待 900秒，即15分钟后再执行下一次策略
