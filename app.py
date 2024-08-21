
# Imports 
import numpy as np 
import pandas as pd 
import yfinance as yf 
import streamlit as st 

import plotly.graph_objects as go

from sklearn.metrics import mean_squared_error, mean_absolute_error
from utils import * 

# Load model 
default_model_path = 'Stock Predictions Model.keras'
model = load_model_from_path(default_model_path)
current_model_path = default_model_path

st.sidebar.header('Model Training & Prediction')
st.markdown("<h1 style='text-align: center; color: white;'>Stock Market Predictor</h1>", unsafe_allow_html=True)

# Downloading Data 
stock = st.text_input('Enter Stock Symbol', 'GOOG')
start_date = st.date_input('Start Date', value=pd.to_datetime('2013-01-01'))
end_date = st.date_input('End Date', value=pd.to_datetime('2023-12-31'))
data = yf.download(stock, start=start_date, end=end_date)

st.subheader('Stock Data (Historical)')
# Displays data 
st.dataframe(data)

# Hyperparameters Selection
units_1, units_2, units_3, dropout_rate, epochs, batch_size = get_hyperparameters()

status_text = st.sidebar.empty()
# Processing and splitting data 
X_train, y_train, X_test, y_test, data, scaler = process_data(data)

# Sidebar Functionalities 
if st.sidebar.button(f'Train New Model using {stock} data', help = f'Click to train a new LSTM model on {stock} data from {start_date} to {end_date}'):
    status_text.text("Training the model, please wait...")
    if X_test is not None: 
        new_model, new_model_path = train_model(X_train, y_train, units_1, units_2, units_3, dropout_rate, epochs)
        status_text.text("Model training complete!") 
        model = new_model 
        current_model_path = new_model_path
    else: 
        status_text.text("Failed to train model")
else: 
    status_text.text("Ready to train a new model!")

# Selecting the MAs to Display 
st.subheader("Select Moving Averages to Display")
ma_options = st.multiselect( label = 'Select MA',
                            options = ['50-Day MA', '100-Day MA', '200-Day MA', 'EMA 50', 'EMA 100'],
                            default = ['50-Day MA', '100-Day MA'])
# Plotting moving averages selection
plot_moving_averages(data, ma_options)

st.subheader("Choose Technical Indicators: ")
plot_rsi = st.checkbox('Relative Strength Index (RSI)')
plot_macd = st.checkbox("Moving Average Convergence Divergence (MACD)")
plot_bollinger = st.checkbox("Bollinger Bands")

# Calculating rsi, macd and bollinger bands 
calc_indicators(data, plot_rsi, plot_macd, plot_bollinger) 

# Plotting rsi, macd and bollinger bands 
plot_indicators(data, plot_rsi, plot_macd, plot_bollinger)

# Predicting prices based on historical data 
predicted_stock_price = st.session_state.model.predict(X_test)
predicted_stock_price = scaler.inverse_transform(predicted_stock_price)
y_test_scaled = scaler.inverse_transform([y_test])

st.subheader('Original Price vs Predicted Price')
fig4 = go.Figure()
fig4.add_trace(go.Scatter(x=list(range(len(y_test_scaled[0]))), y=y_test_scaled[0], mode='lines', name='Original Price', line=dict(color='blue')))
fig4.add_trace(go.Scatter(x=list(range(len(predicted_stock_price))), y=predicted_stock_price.flatten(), mode='lines', name='Predicted Price', line=dict(color='yellow')))
fig4.update_layout(
    xaxis_title='Time',
    yaxis_title='Price',
    height=600,
    width=1200,
)
st.plotly_chart(fig4)

mse = mean_squared_error(y_test_scaled[0], predicted_stock_price)
mae = mean_absolute_error(y_test_scaled[0], predicted_stock_price)
centered_metrics = f"""
<div style = "text-align: center;">
    Mean Squared Error: {mse:.2f} | Mean Absolute Error: {mae: .2f}
</div>
"""
st.markdown(centered_metrics, unsafe_allow_html=True)

# Real-time data comparison (This is ooptional)
st.subheader('Real-Time Data Comparison')

# Fetch real-time data (latest day's data)
real_time_data = yf.download(stock, period='1d', interval='1m')

# Show real-time stock data
st.write("Real-Time Stock Data (Last trading day):")
st.dataframe(real_time_data)

# Plot real-time data
fig5 = go.Figure() 
fig5.add_trace(go.Scatter(
    x = real_time_data.index, 
    y = real_time_data['Close'],
    mode = 'lines', 
    name = 'Real-time Closing Price',
    line = dict(color = 'green')
))
fig5.update_layout(
    title = 'Real-Time Closing Price', 
    xaxis_title = 'Time (minute intervals)', 
    yaxis_title = 'Price', 
    legend_title = 'Legend', 
    template = 'plotly_dark', 
    width = 1200, 
    height = 600
)
st.plotly_chart(fig5)

# Prepare real-time data for prediction (scaling it similarly to historical data)
real_time_close = pd.DataFrame(real_time_data['Close'])
real_time_scaled = scaler.fit_transform(real_time_close.tail(100))

real_time_x = np.array([real_time_scaled])

# Predict based on real-time data
real_time_prediction = st.session_state.model.predict(real_time_x)
real_time_prediction = scaler.inverse_transform(real_time_prediction)

real_time_close_price = f"${real_time_close['Close'][-1]:.2f}"
real_time_prediction_price = f"${real_time_prediction[0][0]:.2f}"

centered_text = f"""
<div style="text-align: center;">
    Real-Time Closing Price: {real_time_close_price} | Real-Time Predicted Closing Price: {real_time_prediction_price}
</div>
"""
st.markdown(centered_text, unsafe_allow_html=True)


