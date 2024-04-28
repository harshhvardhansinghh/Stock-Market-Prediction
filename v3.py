import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
import itertools
from sklearn.linear_model import LinearRegression
import numpy as np


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


n_steps = 100  # Number of future steps to forecast


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



"""
for p in range(0,3):
    for d in range(0,3):
        for q in range(0,3):
            n_series=sensex_data.Close_S
            print(n_series)
            arima_model_n = ARIMA(n_series, order=(p, d, q))
            fitted_model_n = arima_model_n.fit()
            forecast_n = fitted_model_n.forecast(steps=n_steps)
            print(p,d,q)
            print(forecast_n)

"""