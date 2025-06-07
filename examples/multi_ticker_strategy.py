import pandas as pd
from jumpmodels.jump import JumpModel
from jumpmodels.preprocess import DataClipperStd, StandardScalerPD
from jumpmodels.utils.custom_data_utils import prepare_ticker_data

def train_models_for_tickers(multi_ticker_df: pd.DataFrame, feature_cols: list[str], model_params: dict):
    """
    Trains a JumpModel for each ticker in the provided multi-ticker DataFrame.

    Args:
        multi_ticker_df (pd.DataFrame): DataFrame containing data for multiple tickers.
                                        Expected to have a 'ticker' column, a 'date' column,
                                        'closeadj', 'volume', and all columns specified
                                        in `feature_cols`.
        feature_cols (list[str]): A list of column names to be used as features
                                   (e.g., ['nvt', 'pmr', 'taar']).
        model_params (dict): Dictionary of parameters for the JumpModel.
                             Example: {'jump_penalty': 600.0, 'n_components': 2}

    Returns:
        dict: A dictionary where keys are ticker symbols and values are dictionaries
              containing the fitted 'model', 'scaler', and 'clipper' for each ticker.
              Example: {'AAPL': {'model': JumpModel, 'scaler': StandardScalerPD, 'clipper': DataClipperStd}}
    """
    fitted_models_data = {}

    if 'ticker' not in multi_ticker_df.columns:
        print("Warning: 'ticker' column not found in multi_ticker_df. Cannot train models.")
        return fitted_models_data

    unique_tickers = multi_ticker_df['ticker'].unique()

    for ticker_symbol in unique_tickers:
        print(f"Training model for ticker: {ticker_symbol}")
        try:
            X_ticker, ret_ser_ticker, _ = prepare_ticker_data(
                multi_ticker_df, ticker_symbol, feature_cols
            )

            if X_ticker.empty or ret_ser_ticker.empty:
                print(f"Warning: No data available for ticker {ticker_symbol} after preparation. Skipping.")
                continue

            # Initialize preprocessors
            clipper = DataClipperStd(mul=3.)
            scaler = StandardScalerPD()

            # Preprocess data
            X_clipped = clipper.fit_transform(X_ticker)
            X_processed = scaler.fit_transform(X_clipped)

            # Initialize and fit JumpModel
            # Ensure model_params are correctly accessed
            n_components = model_params.get('n_components', 2) # Default to 2 if not provided
            jump_penalty = model_params.get('jump_penalty', 600.0) # Default to 600.0 if not provided

            model = JumpModel(
                n_components=n_components,
                jump_penalty=jump_penalty,
                cont=True
            )
            model.fit(X_processed, ret_ser_ticker, sort_by="cumret")

            # Store fitted model and preprocessors
            fitted_models_data[ticker_symbol] = {
                'model': model,
                'scaler': scaler,
                'clipper': clipper
            }
            print(f"Successfully trained model for ticker: {ticker_symbol}")

        except ValueError as ve:
            print(f"ValueError for ticker {ticker_symbol}: {ve}. Skipping.")
            continue
        except KeyError as ke:
            print(f"KeyError for ticker {ticker_symbol}: {ke}. Check if all required columns "
                  f"('date', 'closeadj', 'volume', and features: {feature_cols}) "
                  f"are present for this ticker. Skipping.")
            continue
        except Exception as e:
            print(f"An unexpected error occurred while training model for ticker {ticker_symbol}: {e}. Skipping.")
            continue

    return fitted_models_data

if __name__ == "__main__":
    # This is where we will load data and call the functions
    print("Multi-ticker strategy script started.")
    # Example:
    # user_df = pd.read_csv("path_to_your_data.csv") # Or however the user loads their data
    # feature_columns = ['nvt', 'pmr', 'taar']
    # params = {'jump_penalty': 600.0, 'n_components': 2}
    # trained_models = train_models_for_tickers(user_df, feature_columns, params)
    # print(f"Trained models for: {list(trained_models.keys())}")

    # --- Configuration ---
    DATA_FILE_PATH = "examples/sample_multiticker_data.csv"
    FEATURE_COLUMNS = ['nvt', 'pmr', 'taar']
    # JumpModel specific parameters. 'cont': True is typical for continuous features in CJM.
    # train_models_for_tickers function will use 'n_components' and 'jump_penalty' from this.
    MODEL_CONFIG = {'jump_penalty': 600.0, 'n_components': 2, 'cont': True}

    # Define train/backtest period boundaries.
    # If None, they will be determined automatically (e.g., midpoint split).
    TRAIN_END_DATE_STR = None  # Example: "2024-01-07"
    BACKTEST_START_DATE_STR = None # Example: "2024-01-08"
    # backtest_end_date will be the last date in the loaded data.

    OUTPUT_PLOT_PATH = "examples/cumulative_return_plot.png"
    # --- End Configuration ---

    # --- Load Data ---
    try:
        raw_df = pd.read_csv(DATA_FILE_PATH)
    except FileNotFoundError:
        print(f"Error: Data file '{DATA_FILE_PATH}' not found.")
        exit()

    # --- Validate Data ---
    if raw_df.empty:
        print(f"Error: Data file '{DATA_FILE_PATH}' is empty.")
        exit()

    raw_df['date'] = pd.to_datetime(raw_df['date'])

    required_cols = FEATURE_COLUMNS + ['date', 'ticker', 'closeadj', 'volume']
    missing_cols = [col for col in required_cols if col not in raw_df.columns]
    if missing_cols:
        print(f"Error: The data file {DATA_FILE_PATH} is missing required columns: {missing_cols}")
        exit()

    # --- Define Training and Backtesting Periods ---
    all_dates = sorted(raw_df['date'].unique())

    if not all_dates:
        print("Error: No unique dates found in the data.")
        exit()

    # Determine train_end_date
    if TRAIN_END_DATE_STR:
        train_end_date = pd.to_datetime(TRAIN_END_DATE_STR)
    else:
        if len(all_dates) < 2: # Need at least 2 dates for a split
             print("Error: Not enough unique dates to perform automatic train/test split.")
             exit()
        train_end_date = all_dates[len(all_dates) // 2]

    # Determine backtest_start_date
    if BACKTEST_START_DATE_STR:
        backtest_start_date = pd.to_datetime(BACKTEST_START_DATE_STR)
    else:
        # Find the first date in all_dates that is greater than train_end_date
        try:
            backtest_start_date = min(d for d in all_dates if d > train_end_date)
        except ValueError: # No date found after train_end_date
            print(f"Error: No data available for backtesting after the training end date: {train_end_date.strftime('%Y-%m-%d')}.")
            exit()

    backtest_end_date = all_dates[-1]

    if backtest_start_date > backtest_end_date:
        print(f"Error: Backtest start date ({backtest_start_date.strftime('%Y-%m-%d')}) "
              f"is after backtest end date ({backtest_end_date.strftime('%Y-%m-%d')}). Check date configurations.")
        exit()

    train_df = raw_df[raw_df['date'] <= train_end_date]

    if train_df.empty:
        print(f"Error: Training data is empty. Check train_end_date: {train_end_date.strftime('%Y-%m-%d')}")
        exit()

    # --- Train Models ---
    print("\n--- Training Models ---")
    print(f"Training data from {train_df['date'].min().strftime('%Y-%m-%d')} to {train_df['date'].max().strftime('%Y-%m-%d')}")
    fitted_models = train_models_for_tickers(train_df, FEATURE_COLUMNS, MODEL_CONFIG)
    print(f"Finished training. Models available for: {list(fitted_models.keys())}")

    if not fitted_models:
        print("No models were trained. Cannot execute strategy.")
        exit()

    # --- Execute Strategy ---
    print("\n--- Executing Trading Strategy ---")
    print(f"Backtesting from {backtest_start_date.strftime('%Y-%m-%d')} to {backtest_end_date.strftime('%Y-%m-%d')}")
    # Pass the full raw_df; execute_trading_strategy will filter by date internally for history
    daily_decisions = execute_trading_strategy(
        raw_df,
        fitted_models,
        FEATURE_COLUMNS,
        backtest_start_date,
        backtest_end_date
    )

    if daily_decisions.empty:
        print("No decisions were made by the strategy. Cannot calculate returns or plot.")
        exit()

    # --- Calculate Returns ---
    print("\n--- Calculating Strategy Returns ---")
    strategy_results_df = calculate_strategy_returns(daily_decisions, raw_df)

    print("\n--- Strategy Results (First 5 & Last 5 days) ---")
    print(strategy_results_df.head())
    print("...")
    print(strategy_results_df.tail())

    if strategy_results_df['cumulative_strategy_return'].empty:
        print("Cumulative strategy return is empty, cannot display final return or plot.")
        exit()

    print("\n--- Final Cumulative Return ---")
    print(strategy_results_df['cumulative_strategy_return'].iloc[-1])

    # --- Plot Results ---
    print("\n--- Plotting Results ---")
    plt.figure(figsize=(12, 7))
    plt.plot(strategy_results_df['date'], strategy_results_df['cumulative_strategy_return'], marker='o', linestyle='-')
    plt.title('Cumulative Strategy Return Over Time')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Return')
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()

    try:
        plt.savefig(OUTPUT_PLOT_PATH)
        print(f"Plot saved to {OUTPUT_PLOT_PATH}")
    except Exception as e:
        print(f"Error saving plot to {OUTPUT_PLOT_PATH}: {e}")


def execute_trading_strategy(
    multi_ticker_df: pd.DataFrame,
    fitted_models_data: dict,
    feature_cols: list[str],
    backtest_start_date: str,
    backtest_end_date: str
):
    """
    Executes a daily trading strategy by selecting the ticker with the highest
    predicted "bull" probability, using volume as a tie-breaker.

    Args:
        multi_ticker_df (pd.DataFrame): DataFrame containing data for multiple tickers.
                                        Must include 'ticker', 'date', 'volume', 'closeadj',
                                        and all columns in `feature_cols`.
        fitted_models_data (dict): Dictionary from `train_models_for_tickers`,
                                   containing fitted 'model', 'scaler', and 'clipper'
                                   for each ticker.
        feature_cols (list[str]): List of feature column names.
        backtest_start_date (str or pd.Timestamp): The start date for the backtesting period.
        backtest_end_date (str or pd.Timestamp): The end date for the backtesting period.

    Returns:
        pd.DataFrame: A DataFrame with columns ['date', 'selected_ticker'], indicating
                      the chosen ticker for each trading day in the backtest period.
    """
    try:
        backtest_start_date = pd.to_datetime(backtest_start_date)
        backtest_end_date = pd.to_datetime(backtest_end_date)
    except ValueError as e:
        print(f"Error: Invalid date format for backtest_start_date or backtest_end_date. {e}")
        return pd.DataFrame(columns=['date', 'selected_ticker'])

    # Ensure multi_ticker_df['date'] is datetime
    if 'date' not in multi_ticker_df.columns:
        print("Error: 'date' column missing from multi_ticker_df.")
        return pd.DataFrame(columns=['date', 'selected_ticker'])

    multi_ticker_df['date'] = pd.to_datetime(multi_ticker_df['date'])

    trading_dates = pd.date_range(start=backtest_start_date, end=backtest_end_date, freq='B')
    daily_decisions = []
    trained_tickers = list(fitted_models_data.keys())

    if not trained_tickers:
        print("Warning: No trained models found in fitted_models_data. Cannot execute strategy.")
        return pd.DataFrame(columns=['date', 'selected_ticker'])

    for current_date in trading_dates:
        print(f"Processing date: {current_date.strftime('%Y-%m-%d')}")
        ticker_predictions_for_date = []

        for ticker_symbol in trained_tickers:
            if ticker_symbol not in fitted_models_data:
                # Should not happen if trained_tickers is from keys, but good for safety
                print(f"Warning: Model data for {ticker_symbol} not found. Skipping for {current_date}.")
                continue

            model_data = fitted_models_data[ticker_symbol]
            model = model_data['model']
            scaler = model_data['scaler']
            clipper = model_data['clipper']

            # Filter data for the current ticker up to and including current_date
            ticker_data_full = multi_ticker_df[multi_ticker_df['ticker'] == ticker_symbol]

            # Ensure data is sorted by date before filtering up to current_date
            # and setting index, crucial for .iloc[-1] later
            ticker_data_up_to_current_date = ticker_data_full[
                ticker_data_full['date'] <= current_date
            ].set_index('date').sort_index()


            if not ticker_data_up_to_current_date.index.isin([current_date]).any():
                # print(f"Data for {ticker_symbol} not available on {current_date.strftime('%Y-%m-%d')}. Skipping.")
                continue

            if not all(col in ticker_data_up_to_current_date.columns for col in feature_cols):
                print(f"Missing some feature columns for {ticker_symbol} on or before {current_date.strftime('%Y-%m-%d')}. Skipping.")
                continue

            X_hist = ticker_data_up_to_current_date[feature_cols]

            if 'volume' not in ticker_data_up_to_current_date.columns:
                 print(f"Volume column missing for {ticker_symbol} on or before {current_date.strftime('%Y-%m-%d')}. Skipping.")
                 continue

            # Ensure current_date is in X_hist for volume lookup and proba extraction
            if current_date not in X_hist.index:
                # This can happen if current_date is a business day but the ticker has no data for it.
                # print(f"Data for {ticker_symbol} does not include the specific date {current_date.strftime('%Y-%m-%d')}. Skipping.")
                continue

            current_volume = ticker_data_up_to_current_date.loc[current_date, 'volume']

            if X_hist.empty:
                # print(f"Feature set X_hist is empty for {ticker_symbol} on {current_date.strftime('%Y-%m-%d')}. Skipping.")
                continue

            try:
                # Important: Use transform, not fit_transform, on historical data with already fitted preprocessors
                X_clipped = clipper.transform(X_hist)
                X_processed = scaler.transform(X_clipped)
            except Exception as e:
                print(f"Error transforming data for {ticker_symbol} on {current_date.strftime('%Y-%m-%d')}: {e}. Skipping.")
                continue

            if X_processed.empty:
                # print(f"X_processed is empty for {ticker_symbol} on {current_date.strftime('%Y-%m-%d')} (possibly too few rows for model). Skipping.")
                continue

            try:
                # predict_proba_online expects enough data to form its internal states if applicable
                proba_df = model.predict_proba_online(X_processed)
            except Exception as e: # Catch specific model prediction errors if known
                print(f"Error predicting probability for {ticker_symbol} on {current_date.strftime('%Y-%m-%d')}: {e}. Skipping.")
                continue

            if proba_df.empty:
                # print(f"Probability DataFrame is empty for {ticker_symbol} on {current_date.strftime('%Y-%m-%d')}. Skipping.")
                continue

            # The last row of proba_df corresponds to the probabilities for current_date
            # State 0 is assumed to be "bull" due to sort_by="cumret" in training
            bull_prob = proba_df.iloc[-1, 0]

            ticker_predictions_for_date.append({
                'ticker': ticker_symbol,
                'bull_prob': bull_prob,
                'volume': current_volume
            })

        if not ticker_predictions_for_date:
            # print(f"No ticker predictions available for {current_date.strftime('%Y-%m-%d')}.")
            continue

        # Sort by bull_prob (descending), then by volume (descending) as tie-breaker
        ticker_predictions_for_date.sort(key=lambda x: (x['bull_prob'], x['volume']), reverse=True)

        selected_ticker_for_the_day = ticker_predictions_for_date[0]['ticker']
        daily_decisions.append({
            'date': current_date,
            'selected_ticker': selected_ticker_for_the_day
        })
        print(f"Date: {current_date.strftime('%Y-%m-%d')}, Selected: {selected_ticker_for_the_day} "
              f"(Prob: {ticker_predictions_for_date[0]['bull_prob']:.4f}, Vol: {ticker_predictions_for_date[0]['volume']})")


    if not daily_decisions:
        print("No daily decisions were made during the backtest period.")
        return pd.DataFrame(columns=['date', 'selected_ticker'])

    return pd.DataFrame(daily_decisions)


def calculate_strategy_returns(
    daily_decisions_df: pd.DataFrame,
    multi_ticker_df: pd.DataFrame
):
    """
    Calculates daily and cumulative returns for the trading strategy based on
    selected tickers and their next-day closing prices.

    Args:
        daily_decisions_df (pd.DataFrame): DataFrame from `execute_trading_strategy`,
                                           containing 'date' and 'selected_ticker'.
                                           'date' should be the date of decision/trade.
        multi_ticker_df (pd.DataFrame): Original DataFrame with historical data for all tickers.
                                        Must include 'date', 'ticker', and 'closeadj' columns.

    Returns:
        pd.DataFrame: The `daily_decisions_df` augmented with 'next_day_return' and
                      'cumulative_strategy_return' columns.
    """
    if 'date' not in daily_decisions_df.columns or 'selected_ticker' not in daily_decisions_df.columns:
        print("Error: daily_decisions_df must contain 'date' and 'selected_ticker' columns.")
        return daily_decisions_df # Or raise error

    if not {'date', 'ticker', 'closeadj'}.issubset(multi_ticker_df.columns):
        print("Error: multi_ticker_df must contain 'date', 'ticker', and 'closeadj' columns.")
        return daily_decisions_df # Or raise error

    # Ensure date columns are datetime objects
    daily_decisions_df['date'] = pd.to_datetime(daily_decisions_df['date'])
    multi_ticker_df['date'] = pd.to_datetime(multi_ticker_df['date'])

    # Prepare multi_ticker_df for efficient lookup
    data_indexed = multi_ticker_df.set_index(['date', 'ticker'])
    data_indexed.sort_index(inplace=True)

    returns_list = []

    for row in daily_decisions_df.itertuples():
        trade_date = row.date # This is already a Timestamp if converted above
        selected_ticker = row.selected_ticker
        next_trading_day = trade_date + pd.tseries.offsets.BDay(1)

        daily_return = 0.0 # Default to 0% return in case of issues

        try:
            close_trade_date = data_indexed.loc[(trade_date, selected_ticker), 'closeadj']

            # Check if next_trading_day exists for the ticker
            if (next_trading_day, selected_ticker) in data_indexed.index:
                close_next_day = data_indexed.loc[(next_trading_day, selected_ticker), 'closeadj']

                if pd.notna(close_trade_date) and pd.notna(close_next_day) and close_trade_date > 0:
                    daily_return = (close_next_day - close_trade_date) / close_trade_date
                else:
                    print(f"Warning: Invalid price data for {selected_ticker} around {trade_date.strftime('%Y-%m-%d')}. "
                          f"Close on trade date: {close_trade_date}, Close on next day: {close_next_day}. Assigning 0% return.")
            else:
                print(f"Warning: Next trading day data ({next_trading_day.strftime('%Y-%m-%d')}) not found for {selected_ticker}. "
                      f"Trade date: {trade_date.strftime('%Y-%m-%d')}. Assigning 0% return.")

        except KeyError:
            print(f"Warning: Price data not found for {selected_ticker} on {trade_date.strftime('%Y-%m-%d')} or "
                  f"{next_trading_day.strftime('%Y-%m-%d')}. Assigning 0% return.")
        except Exception as e:
            print(f"An unexpected error occurred calculating returns for {selected_ticker} on {trade_date.strftime('%Y-%m-%d')}: {e}. Assigning 0% return.")

        returns_list.append(daily_return)

    daily_decisions_df['next_day_return'] = returns_list

    # Calculate cumulative returns
    # Ensure no NaNs are accidentally introduced if 0.0 was strictly used.
    # If np.nan was used, then skipna=True would be important in cumprod, or fillna(0) before.
    daily_decisions_df['cumulative_strategy_return'] = (1 + daily_decisions_df['next_day_return']).cumprod() - 1

    return daily_decisions_df
