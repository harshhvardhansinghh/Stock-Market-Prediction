from flask import Flask, render_template, request, jsonify
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
from io import BytesIO
import base64
from nsepy import get_history
from datetime import datetime, timedelta
import yfinance as yf

app = Flask(__name__)

def get_latest_values():
    usd_inr = yf.Ticker('USDINR=X').history(period='1d')['Close'].iloc[-1]
    nifty = yf.Ticker('^NSEI').history(period='1d')['Close'].iloc[-1]
    sensex = yf.Ticker('^BSESN').history(period='1d')['Close'].iloc[-1]

    return usd_inr, nifty, sensex

# Your existing code for loading and preprocessing data here...
p_values = range(5)
d_values = range(5)
q_values = range(5)

# Load data
nifty_data = pd.read_csv('nifty_data.csv')
sensex_data = pd.read_csv('sensex_data.csv')
usd_inr_data = pd.read_csv('usd_inr_data.csv')

# Merge dataframes
nifty_inr = usd_inr_data.merge(nifty_data, how='outer', on='Date')
sensex_inr = usd_inr_data.merge(sensex_data, how='outer', on='Date')

# Drop NaN values
nifty_inr.dropna(inplace=True)
sensex_inr.dropna(inplace=True)

# Replace ',' and convert to float for NIFTY DataFrame
nifty_inr['Close_N'] = nifty_inr['Close_N'].replace(',', '', regex=True).astype(float)
nifty_inr['Open_N'] = nifty_inr['Open_N'].replace(',', '', regex=True).astype(float)
nifty_inr['Low_N'] = nifty_inr['Low_N'].replace(',', '', regex=True).astype(float)
nifty_inr['High_N'] = nifty_inr['High_N'].replace(',', '', regex=True).astype(float)

# Convert 'Date' to datetime
nifty_inr['Date'] = pd.to_datetime(nifty_inr['Date'])

# Set 'Date' as the index
nifty_inr.set_index('Date', inplace=True)

# Split data into training and testing sets
train_size = int(len(nifty_inr) * 0.15)
train_data = nifty_inr.iloc[train_size:]
test_data = nifty_inr.iloc[:train_size]

@app.route('/')
def index():
    # Getting the latest values
    usd_inr, nifty, sensex = get_latest_values()

    return render_template('index.html', usd_inr=usd_inr, nifty=nifty, sensex=sensex)

@app.route('/generate_plots', methods=['POST'])
def generate_plots():
    global sensex_inr
    usd_inr, nifty, sensex = get_latest_values() 
    input_days = int(request.form['input_days'])

    # Your existing code for ARIMA predictions here...
    # ...
    train_endog = train_data[['Close_N']]
    train_exog = train_data[['Open', 'High', 'Low', 'Close', 'Open_N', 'High_N', 'Low_N']]

    test_endog = test_data[['Close_N']]
    test_exog = test_data[['Open', 'High', 'Low', 'Close', 'Open_N', 'High_N', 'Low_N']]

    # Convert columns to numeric format
    train_endog = train_endog.apply(pd.to_numeric, errors='coerce')
    train_exog = train_exog.apply(pd.to_numeric, errors='coerce')
    test_endog = test_endog.apply(pd.to_numeric, errors='coerce')
    test_exog = test_exog.apply(pd.to_numeric, errors='coerce')

    # Drop rows with NaN values after conversion
    train_endog.dropna(inplace=True)
    train_exog.dropna(inplace=True)
    test_endog.dropna(inplace=True)
    test_exog.dropna(inplace=True)

    # Grid search for the best ARIMA order for NIFTY
    best_mae_nifty = float('inf')  # Initialize with a large value
    best_order_nifty = None

    # Try different values for p, d, q
    endog_column_nifty = train_endog.iloc[:, 0]

    model_nifty = ARIMA(endog_column_nifty, exog=train_exog, order=(0, 0, 4))
    results_nifty = model_nifty.fit()

    forecast_nifty = results_nifty.get_forecast(steps=len(test_endog), exog=test_exog)
    predicted_values_nifty = forecast_nifty.predicted_mean
    mae_nifty = mean_absolute_error(test_endog['Close_N'], predicted_values_nifty)
    best_order_nifty = (1, 3, 3)

    print(f"Best ARIMA Order for NIFTY: {best_order_nifty}")
    print(f"Mean Absolute Error of NIFTY: {mae_nifty}")
    # Plot NIFTY
    plt.figure(figsize=(10, 6))
    plt.plot(test_data.index, test_data['Close_N'], label='Actual NIFTY Close')
    plt.plot(test_data.index, predicted_values_nifty, label='Predicted NIFTY Close', linestyle='--')
    plt.xlabel('Date')
    plt.ylabel('NIFTY Close')
    plt.title('Actual vs Predicted NIFTY Close')
    plt.legend()
    plt.xticks(rotation=45)
    nifty_plot = get_image()

    sensex_inr['Close_S'] = sensex_inr['Close_S'].replace(',', '', regex=True).astype(float)
    sensex_inr['Open_S'] = sensex_inr['Open_S'].replace(',', '', regex=True).astype(float)
    sensex_inr['Low_S'] = sensex_inr['Low_S'].replace(',', '', regex=True).astype(float)
    sensex_inr['High_S'] = sensex_inr['High_S'].replace(',', '', regex=True).astype(float)

    # Convert 'Date' to datetime
    sensex_inr['Date'] = pd.to_datetime(sensex_inr['Date'])

    # Set 'Date' as the index
    sensex_inr.set_index('Date', inplace=True)

    # Split data into training and testing sets
    train_size_sensex = int(len(sensex_inr) * 0.15)
    train_data_sensex = sensex_inr.iloc[train_size_sensex:]
    test_data_sensex = sensex_inr.iloc[:train_size_sensex]

    # Separate endogenous and exogenous variables for training and testing
    train_endog_sensex = train_data_sensex[['Close_S']]
    train_exog_sensex = train_data_sensex[['Open', 'High', 'Low', 'Close', 'Open_S', 'High_S', 'Low_S']]

    test_endog_sensex = test_data_sensex[['Close_S']]
    test_exog_sensex = test_data_sensex[['Open', 'High', 'Low', 'Close', 'Open_S', 'High_S', 'Low_S']]

    # Convert columns to numeric format
    train_endog_sensex = train_endog_sensex.apply(pd.to_numeric, errors='coerce')
    train_exog_sensex = train_exog_sensex.apply(pd.to_numeric, errors='coerce')
    test_endog_sensex = test_endog_sensex.apply(pd.to_numeric, errors='coerce')
    test_exog_sensex = test_exog_sensex.apply(pd.to_numeric, errors='coerce')

    # Drop rows with NaN values after conversion
    train_endog_sensex.dropna(inplace=True)
    train_exog_sensex.dropna(inplace=True)
    test_endog_sensex.dropna(inplace=True)
    test_exog_sensex.dropna(inplace=True)

    # Grid search for the best ARIMA order for Sensex
    best_mae_sensex = float('inf')  # Initialize with a large value

    # Try different values for p, d, q
    endog_column_sensex = train_endog_sensex.iloc[:, 0]
    model_sensex = ARIMA(endog_column_sensex, exog=train_exog_sensex, order=(0, 0, 3))
    results_sensex = model_sensex.fit()

    forecast_sensex = results_sensex.get_forecast(steps=len(test_endog_sensex), exog=test_exog_sensex)
    predicted_values_sensex = forecast_sensex.predicted_mean
    mae_sensex = mean_absolute_error(test_endog_sensex['Close_S'], predicted_values_sensex)

    best_order_sensex = (0, 0, 3)
    print(f"Best ARIMA Order for Sensex: {best_order_sensex}")
    print(f"Mean Absolute Error of Sensex: {mae_sensex}")

    # Plot Sensex
    plt.figure(figsize=(10, 6))
    plt.plot(test_data_sensex.index, test_data_sensex['Close_S'], label='Actual Sensex Close')
    plt.plot(test_data_sensex.index, predicted_values_sensex, label='Predicted Sensex Close', linestyle='--')
    plt.xlabel('Date')
    plt.ylabel('Sensex Close')
    plt.title('Actual vs Predicted Sensex Close')
    plt.legend()
    plt.xticks(rotation=45)
    sensex_plot = get_image()


    nifty_data = pd.read_csv('nifty_data_reverse.csv')
    sensex_data = pd.read_csv('sensex_data_reverse.csv')
    usd_inr_data = pd.read_csv('usd_inr_data_reverse.csv')

    # Merge dataframes
    nifty_inr = usd_inr_data.merge(nifty_data, how='outer', on='Date')
    sensex_inr = usd_inr_data.merge(sensex_data, how='outer', on='Date')

    # Drop NaN values
    nifty_inr.dropna(inplace=True)
    sensex_inr.dropna(inplace=True)

    # Replace ',' and convert to float for NIFTY DataFrame
    nifty_inr['Close_N'] = nifty_inr['Close_N'].replace(',', '', regex=True).astype(float)
    nifty_inr['Open_N'] = nifty_inr['Open_N'].replace(',', '', regex=True).astype(float)
    nifty_inr['Low_N'] = nifty_inr['Low_N'].replace(',', '', regex=True).astype(float)
    nifty_inr['High_N'] = nifty_inr['High_N'].replace(',', '', regex=True).astype(float)

    # Convert 'Date' to datetime
    nifty_inr['Date'] = pd.to_datetime(nifty_inr['Date'])

    # Set 'Date' as the index
    #nifty_inr.set_index('Date', inplace=True)

    # Separate endogenous and exogenous variables for training and testing
    train_endog_nifty = nifty_inr[['Close_N']]
    train_exog_nifty = nifty_inr[[ 'Open','High', 'Low', 'Close','Open_N', 'High_N', 'Low_N']]

    # Convert columns to numeric format
    train_endog_nifty = train_endog_nifty.apply(pd.to_numeric, errors='coerce')
    train_exog_nifty = train_exog_nifty.apply(pd.to_numeric, errors='coerce')

    # Drop rows with NaN values after conversion
    train_endog_nifty.dropna(inplace=True)
    train_exog_nifty.dropna(inplace=True)

    # Replace ',' and convert to float for Sensex DataFrame
    sensex_inr['Close_S'] = sensex_inr['Close_S'].replace(',', '', regex=True).astype(float)
    sensex_inr['Open_S'] = sensex_inr['Open_S'].replace(',', '', regex=True).astype(float)
    sensex_inr['Low_S'] = sensex_inr['Low_S'].replace(',', '', regex=True).astype(float)
    sensex_inr['High_S'] = sensex_inr['High_S'].replace(',', '', regex=True).astype(float)

    # Convert 'Date' to datetime
    sensex_inr['Date'] = pd.to_datetime(sensex_inr['Date'])

    # Set 'Date' as the index
    #sensex_inr.set_index('Date', inplace=True)

    # Separate endogenous and exogenous variables for training and testing
    train_endog_sensex = sensex_inr[[ 'Close_S']]
    train_exog_sensex = sensex_inr[['Open', 'High', 'Low', 'Close','Open_S', 'Low_S']]

    # Convert columns to numeric format
    train_endog_sensex = train_endog_sensex.apply(pd.to_numeric, errors='coerce')
    train_exog_sensex = train_exog_sensex.apply(pd.to_numeric, errors='coerce')

    # Drop rows with NaN values after conversion
    train_endog_sensex.dropna(inplace=True)
    train_exog_sensex.dropna(inplace=True)

    input_days = int(request.form['input_days'])
    n_steps = input_days  # Number of future steps to forecast


    #######################################################################################################


    close_series=nifty_inr.Close

    arima_model_c = ARIMA(close_series, order=(1, 2, 2))
    fitted_model_c = arima_model_c.fit()

    # After fitting your ARIMA model, forecast future values
    forecast_c = fitted_model_c.forecast(steps=n_steps)

    # Print the forecasted values
    #print(forecast_c)

    #######################################################################################################

    open_series=nifty_inr.Open

    arima_model_o = ARIMA(open_series, order=(1, 2, 2))
    fitted_model_o = arima_model_o.fit()

    # After fitting your ARIMA model, forecast future values
    forecast_o = fitted_model_o.forecast(steps=n_steps)

    # Print the forecasted values
    #print(forecast_o)


    #######################################################################################################

    high_series=nifty_inr.High

    arima_model_h = ARIMA(high_series, order=(1, 2, 2))
    fitted_model_h = arima_model_h.fit()

    # After fitting your ARIMA model, forecast future values
    forecast_h = fitted_model_h.forecast(steps=n_steps)

    # Print the forecasted values
    #print(forecast_h)


    #######################################################################################################

    low_series=nifty_inr.Low

    arima_model_l = ARIMA(low_series, order=(1, 2, 2))
    fitted_model_l = arima_model_l.fit()

    # After fitting your ARIMA model, forecast future values
    forecast_l = fitted_model_l.forecast(steps=n_steps)

    # Print the forecasted values
    #print(forecast_l)

    #######################################################################################################





    n_series=nifty_inr.Open_N
    arima_model_n_o = ARIMA(n_series, order=(2, 2, 2))
    fitted_model_n_o = arima_model_n_o.fit()
    forecast_n_o = fitted_model_n_o.forecast(steps=n_steps)
    #print(forecast_n_o)


    n_series=nifty_inr.High_N
    arima_model_n_h = ARIMA(n_series, order=(2, 2, 2))
    fitted_model_n_h = arima_model_n_h.fit()
    forecast_n_h = fitted_model_n_h.forecast(steps=n_steps)
    #print(forecast_n_h)

    n_series=nifty_inr.Low_N
    arima_model_n_l = ARIMA(n_series, order=(2, 2, 2))
    fitted_model_n_l = arima_model_n_l.fit()
    forecast_n_l = fitted_model_n_l.forecast(steps=n_steps)
    #print(forecast_n_l)


    nifty_inr_extended = {
        'Open': forecast_o,
        'Close': forecast_c,
        'High': forecast_h,
        'Low': forecast_l,
        'Open_N': forecast_n_o ,
        'High_N': forecast_n_h ,
        'Low_N' : forecast_n_l
    }

    # Converting dictionary to DataFrame
    df = pd.DataFrame(nifty_inr_extended)

    #print(nifty_inr_extended)

    print(df)





    ################################################################################################3


    s_series=train_exog_sensex.Open_S
    arima_model_s_o = ARIMA(s_series, order=(2, 2, 2))
    fitted_model_s_o = arima_model_s_o.fit()
    forecast_s_o = fitted_model_s_o.forecast(steps=n_steps)
    #print(forecast_s_o)


    s_series=train_exog_sensex.High
    arima_model_s_h = ARIMA(n_series, order=(2, 2, 2))
    fitted_model_s_h = arima_model_s_h.fit()
    forecast_s_h = fitted_model_s_h.forecast(steps=n_steps)
    #print(forecast_s_h)

    s_series=train_exog_sensex.Low_S
    arima_model_s_l = ARIMA(s_series, order=(2, 2, 2))
    fitted_model_s_l = arima_model_s_l.fit()
    forecast_s_l = fitted_model_s_l.forecast(steps=n_steps)
    #print(forecast_s_l)



    sensex_inr_extended = {
        'Open': forecast_o,
        'Close': forecast_c,
        'High': forecast_h,
        'Low': forecast_l,
        'Open_S': forecast_s_o ,
        'Low_S' : forecast_s_l

    }

    # Converting dictionary to DataFrame
    dfx = pd.DataFrame(sensex_inr_extended)

    dfx.drop(dfx.index[0], inplace=True)

    dfs = dfx.fillna(method='ffill')
    #print(sensex_inr_extended)

    print(dfs)




    ################################################################################################




    endog_column_nifty = train_endog_nifty.iloc[:, 0]
    model_nifty = ARIMA(endog_column_nifty, exog=train_exog_nifty, order=(0, 0, 4))
    results_nifty = model_nifty.fit()

    forecast_nifty = results_nifty.get_forecast(steps=n_steps, exog=df)
    predicted_values_nifty = forecast_nifty.predicted_mean

    print(predicted_values_nifty)

    best_order_nifty = (0, 0, 4)



    endog_column_sensex = train_endog_sensex.iloc[:, 0]
    model_sensex = ARIMA(endog_column_sensex, exog=train_exog_sensex, order=(0, 0, 3))
    results_sensex = model_sensex.fit()

    forecast_sensex = results_sensex.get_forecast(steps=n_steps, exog=dfs)
    predicted_values_sensex = forecast_sensex.predicted_mean

    print(predicted_values_sensex)

    best_order_sensex = (0, 0, 3)
    predicted_values_nifty = list(predicted_values_nifty)  # Convert to a list
    predicted_values_sensex = list(predicted_values_sensex)  # Convert to a list

    print("NIFTY Forecast:", predicted_values_nifty)
    print("SENSEX Forecast:", predicted_values_sensex)

    # Pass the data to the template
    return render_template('index.html', usd_inr=usd_inr, nifty=nifty, sensex=sensex, nifty_plot=nifty_plot, sensex_plot=sensex_plot, nifty_forecast=predicted_values_nifty, sensex_forecast=predicted_values_sensex)

def get_image():
    img = BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()
    return f'data:image/png;base64,{plot_url}'

if __name__ == '__main__':
    app.run(debug=True)
