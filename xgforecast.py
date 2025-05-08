    import yfinance as yf
    import pandas as pd
    import numpy as np
    from xgboost import XGBRegressor
    from sklearn.model_selection import TimeSeriesSplit, cross_val_score
    from sklearn.metrics import mean_squared_error
    from fredapi import Fred
    import matplotlib.pyplot as plt
    from sklearn.metrics import r2_score

    # Set your FRED API Key
    FRED_API_KEY = '91a4916417280b4ea35b32dee2d2a82b'   # Or paste your API key directly
    fred = Fred(api_key=FRED_API_KEY)

    # Tickers and interest rate
    tickers = ['XLK', 'XLF', 'XLU', 'XLY', 'XLP', 'XLE', 'XLI', 'XLV', 'XLRE', 'XLB', 'XLC']
    interest_rate_series = 'GS3M'  # 10-Year Treasury Constant Maturity Rate

    # Download ETF data
    etf_data = yf.download(tickers, start='1999-11-01', end='2024-12-31', auto_adjust=True)['Close']
    returns = etf_data.pct_change().dropna()

    # Download interest rates
    rates = fred.get_series(interest_rate_series).resample('B').ffill()
    rates.name = 'interest_rate'
    rates = rates.pct_change().dropna()  # Use rate change, not level

    # Combine data
    combined = returns.join(rates, how='inner').dropna()

    # Feature engineering: include lagged returns and rate changes
    lags = 3
    for ticker in tickers:
        for lag in range(1, lags + 1):
            combined[f'{ticker}_lag{lag}'] = combined[ticker].shift(lag)

    for lag in range(1, lags + 1):
        combined[f'rate_lag{lag}'] = combined['interest_rate'].shift(lag)

    combined.dropna(inplace=True)

    # Forecasting one ETF (repeat for all tickers)

    forecast_results = {}


    for stock in tickers:
        target_ticker = stock
        X = combined.drop(columns=tickers + ['interest_rate'])
        y = combined[target_ticker]

        # Train-test split
        train_size = int(0.8 * len(X))
        X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
        y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]

        # Model training
        model = XGBRegressor(n_estimators=100, max_depth=4, learning_rate=0.05)
        model.fit(X_train, y_train)

        # Prediction
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        print(f'MSE for {target_ticker}: {mse:.6f}')

        r2 = r2_score(y_test, y_pred)
        print(f'R² Score: {r2:.4f}')

        #save results
        forecast_results[stock] = {'y_pred' : y_pred, 'mse' : mse, 'r2' : r2}

        # Plot actual vs predicted
        plt.figure(figsize=(10,5))
        plt.plot(y_test.index, y_test.values, label='Actual')
        plt.plot(y_test.index, y_pred, label='Predicted')
        plt.title(f'{target_ticker} - Actual vs Predicted Returns')
        plt.legend()
        plt.show()
    
