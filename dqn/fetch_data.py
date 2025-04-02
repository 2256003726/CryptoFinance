from binance.client import Client
import pandas as pd

def fetch_data(symbol="BNBUSDT", interval=Client.KLINE_INTERVAL_1HOUR, limit=1000, api_key=None, api_secret=None):
    """
    从 Binance 获取指定交易对的 K 线数据
    :param symbol: 交易对（默认 "BNBUSDT"）
    :param interval: K 线周期（默认 1 小时）
    :param limit: 获取的 K 线数量（最大 1000）
    :param api_key: Binance API Key（可选）
    :param api_secret: Binance API Secret（可选）
    :return: DataFrame 格式的 OHLCV 数据
    """
    # 连接 Binance API（如果提供了 API Key）
    client = Client(api_key, api_secret) if api_key else Client()

    # 获取 K 线数据
    ohlcv = client.get_klines(symbol=symbol, interval=interval, limit=limit)

    # 转换为 DataFrame
    df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume",
                                      "close_time", "quote_asset_volume", "num_trades",
                                      "taker_buy_base", "taker_buy_quote", "ignore"])

    # 选择需要的列

    df = df[["timestamp", "open", "high", "low", "close", "volume",
         "quote_asset_volume", "num_trades", "taker_buy_base", "taker_buy_quote"]]


    # 时间戳转换为 datetime 格式
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")

    # 设置索引
    df.set_index("timestamp", inplace=True)

    # 数据转换为浮点数，并保留 4 位小数
    df = df.astype(float).round(4)

    return df

# 示例：获取 BNB/USDT 1 小时 K 线数据
df = fetch_data(symbol="BNBUSDT", interval=Client.KLINE_INTERVAL_1HOUR, limit=1000)
print(df.head())

# 保存为 CSV
df.to_csv("crypto_data.csv")
print("数据已保存到 crypto_data.csv")
