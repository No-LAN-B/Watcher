from abc import ABC, abstractmethod
import pandas as pd

class DataProvider(ABC):
    @abstractmethod
    def fetch(self,
              tickers: list[str],
              start: str,
              end: str) -> dict[str, pd.DataFrame]:
        #OHLC = Open High Low Close Chart
        """
        Fetch historical OHLC+Volume data for each ticker. 
        Returns a dict mapping ticker → DataFrame.
        """
        pass

import yfinance as yf

class YFinanceProvider(DataProvider):
    def fetch(self,
              tickers: list[str],
              start: str,
              end: str) -> dict[str, pd.DataFrame]:
        data: dict[str, pd.DataFrame] = {}
        for ticker in tickers:
            df = yf.download(ticker, start=start, end=end)
            # collapse to single level (take the second level—your actual fields)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)  # ["Open","High","Low",…]
            df.index = pd.to_datetime(df.index)
            data[ticker] = df
        return data
