import pandas as pd
from binance.client import Client
from sklearn.preprocessing import MinMaxScaler

import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)


def fetch_data(symbol="BNBUSDT", interval=Client.KLINE_INTERVAL_1HOUR, limit=10000):
    client = Client()
    ohlcv = client.get_klines(symbol=symbol, interval=interval, limit=limit)
    df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume", "close_time",
                                      "quote_asset_volume", "num_trades", "taker_buy_base", "taker_buy_quote",
                                      "ignore"])
    df = df[["timestamp", "open", "high", "low", "close", "volume"]]
    # 将 timestamp 列转换为 datetime 格式
    df.loc[:, "timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df.set_index("timestamp", inplace=True)
    # 保留两位小数
    df[["open", "high", "low", "close", "volume"]] = df[["open", "high", "low", "close", "volume"]].astype(float).round(4)
    return df


def preprocess_data(df, features=None, seq_length=48):
    if features is None:
        features = ["open", "high", "low", "close", "volume"]
    scaler = MinMaxScaler()
    df_scaled = scaler.fit_transform(df[features])
    return df_scaled, scaler


if __name__ == "__main__":
    df1 = fetch_data()
    df_scaled1, scaler1 = preprocess_data(df1)
    print(df1.head())
