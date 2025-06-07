import pandas as pd

def prepare_ticker_data(df: pd.DataFrame, ticker_symbol: str, feature_cols: list[str]):
    """
    Prepares and filters data for a specific ticker symbol from a multi-ticker DataFrame.

    Args:
        df (pd.DataFrame): The input DataFrame containing data for multiple tickers.
                           It is expected to have 'ticker', 'date', 'volume', 'closeadj',
                           and all columns specified in `feature_cols`.
        ticker_symbol (str): The ticker symbol to filter the DataFrame by (e.g., 'AAPL').
        feature_cols (list[str]): A list of column names to be selected as features
                                   (e.g., ['nvt', 'pmr', 'taar']).

    Returns:
        tuple: A tuple containing three pandas objects:
            - X_ticker (pd.DataFrame): DataFrame with selected feature columns, indexed by date.
            - ret_ser_ticker (pd.Series): Series of daily percentage returns, indexed by date.
            - volume_ticker (pd.Series): Series of trading volume, indexed by date.

    Raises:
        ValueError: If `ticker_symbol` is not found in the 'ticker' column of the DataFrame.
        KeyError: If 'date', 'ticker', 'volume', 'closeadj' or any of the `feature_cols`
                  are not present in the DataFrame columns.
    """
    # Create a copy to avoid modifying the original DataFrame
    df_copy = df.copy()

    # Filter by ticker symbol
    ticker_df = df_copy[df_copy['ticker'] == ticker_symbol]

    if ticker_df.empty:
        raise ValueError(f"Ticker symbol '{ticker_symbol}' not found in the DataFrame.")

    # Set 'date' column as index and ensure it's datetime type
    if 'date' not in ticker_df.columns:
        raise KeyError("The 'date' column is missing from the DataFrame.")
    ticker_df['date'] = pd.to_datetime(ticker_df['date'])
    ticker_df = ticker_df.set_index('date')
    ticker_df = ticker_df.sort_index() # Ensure chronological order

    # Select feature columns
    missing_features = [col for col in feature_cols if col not in ticker_df.columns]
    if missing_features:
        raise KeyError(f"The following feature columns are missing: {missing_features}")
    X_ticker = ticker_df[feature_cols]

    # Extract volume
    if 'volume' not in ticker_df.columns:
        raise KeyError("The 'volume' column is missing from the DataFrame.")
    volume_ticker = ticker_df['volume']

    # Calculate returns from 'closeadj'
    if 'closeadj' not in ticker_df.columns:
        raise KeyError("The 'closeadj' column is missing from the DataFrame.")
    closeadj_series = ticker_df['closeadj']
    ret_ser_ticker = closeadj_series.pct_change()

    # Handle potential NaN in the first row of returns series
    if not ret_ser_ticker.empty and pd.isna(ret_ser_ticker.iloc[0]):
        ret_ser_ticker.iloc[0] = 0.0
    # Alternatively, to drop NaNs (if preferred):
    # ret_ser_ticker = ret_ser_ticker.dropna()

    return X_ticker, ret_ser_ticker, volume_ticker
