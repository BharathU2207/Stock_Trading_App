from keras.models import Sequential, load_model
from keras.layers import Dense, LSTM, Dropout 
import pandas_ta as ta 
import numpy as np 
import pandas as pd 
import streamlit as st 
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler
from tqdm.keras import TqdmCallback

def load_model_from_path(default_model_path):
    if 'model' not in st.session_state:
        st.session_state.model = load_model(default_model_path)
    return st.session_state.model

def get_hyperparameters():
    units_1 = st.sidebar.slider('Units in 1st LSTM Layer', 10, 200, 50, 10)
    units_2 = st.sidebar.slider('Units in 2nd LSTM Layer', 10, 200, 60, 10)
    units_3 = st.sidebar.slider('Units in 3rd LSTM Layer', 10, 200, 80, 10)
    dropout_rate = st.sidebar.slider('Dropout Rate', 0.0, 0.5, 0.2, 0.1)
    epochs = st.sidebar.slider('Number of Epochs', 1, 50, 10)
    batch_size = st.sidebar.slider('Batch Size', 16, 128, 32, 16)

    return units_1, units_2 , units_3, dropout_rate, epochs, batch_size

def train_model(X_train, y_train, units_1, units_2, units_3, dropout_rate, epochs):
    # Progress bar for model training
    progress_bar = st.sidebar.progress(0)
    new_model = build_model(units_1, units_2, units_3, dropout_rate, X_train)
    for epoch in range(epochs):
        new_model.fit(X_train, y_train, epochs=1, batch_size=32, verbose=0, callbacks=[TqdmCallback(verbose=1)])
        progress_bar.progress((epoch + 1) / epochs)
 
    # Save the new model
    new_model_path = r'C:\Users\bhara\Desktop\Projects\StockNet\Stock Predictions user trained Model.keras'
    new_model.save(new_model_path)
    st.session_state.model = new_model
    return new_model, new_model_path

def plot_moving_averages(data, ma_options): 
    ma_50_days, ma_100_days, ma_200_days, ema_50, ema_100 = options(ma_options, data)
    st.subheader('Price with Selected Moving Averages')
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data.index, y=data['Close'], mode='lines', name='Original Price', line=dict(color='blue')))
    if ma_50_days is not None:
        fig.add_trace(go.Scatter(x=ma_50_days.index, y=ma_50_days, mode='lines', name='50-Day MA', line=dict(color='yellow')))
    if ma_100_days is not None:
        fig.add_trace(go.Scatter(x=ma_100_days.index, y=ma_100_days, mode='lines', name='100-Day MA', line=dict(color='pink')))
    if ma_200_days is not None:
        fig.add_trace(go.Scatter(x=ma_200_days.index, y=ma_200_days, mode='lines', name='200-Day MA', line=dict(color='red')))
    if ema_50 is not None:
        fig.add_trace(go.Scatter(x=ema_50.index, y=ema_50, mode='lines', name='EMA 50', line=dict(color='green')))
    if ema_100 is not None: 
        fig.add_trace(go.Scatter(x=ema_100.index, y=ema_100, mode='lines', name='EMA 100', line=dict(color='orange')))
    fig.update_layout(
        xaxis_title='Date',
        yaxis_title='Price',
        legend_title='Legend',
        height=600,
        width=1200,
    )
    st.plotly_chart(fig)
    
def create_dataset(dataset, look_back=100):
    X, y = [], []
    for i in range(look_back, len(dataset)):
        X.append(dataset[i-look_back:i, 0])
        y.append(dataset[i, 0])
    return np.array(X), np.array(y)

def process_data(data): 
    data.reset_index(inplace = True)  
    data.dropna(inplace=True)

    data_train = pd.DataFrame(data.Close[0: int(len(data)* 0.80)])
    data_test = pd.DataFrame(data.Close[int(len(data)*0.80): len(data)])

    scaler = MinMaxScaler(feature_range = (0,1))
    data_train_scaled = scaler.fit_transform(data_train)
    data_test_scaled = scaler.fit_transform(data_test)

    X_train, y_train = create_dataset(data_train_scaled)
    X_test, y_test = create_dataset(data_test_scaled)

    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    if X_test.size > 0:
        X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    else:
        st.write("Not enough data to create a valid test set. Please select a longer date range or decrease look_back.")
    
    return X_train, y_train, X_test, y_test, data, scaler

def build_model(units_1, units_2, units_3, dropout_rate, X_train):
        model = Sequential()
        model.add(LSTM(units=units_1, activation='relu', return_sequences=True, input_shape=(X_train.shape[1], 1)))
        model.add(Dropout(dropout_rate))
        model.add(LSTM(units=units_2, activation='relu', return_sequences=True))
        model.add(Dropout(dropout_rate))
        model.add(LSTM(units=units_3, activation='relu', return_sequences=True))
        model.add(LSTM(units = 120, activation = 'relu', return_sequences = False))
        model.add(Dropout(dropout_rate))
        model.add(Dense(units=1))

        model.compile(optimizer='adam', loss='mean_squared_error')
        return model


def options(ma_options, data):
        if '50-Day MA' in ma_options:
            ma_50_days = data['Close'].rolling(50).mean()
        else:
            ma_50_days = None

        if '100-Day MA' in ma_options:
            ma_100_days = data['Close'].rolling(100).mean()
        else:
            ma_100_days = None

        if '200-Day MA' in ma_options:
            ma_200_days = data['Close'].rolling(200).mean()
        else:
            ma_200_days = None

        if 'EMA 50' in ma_options:
            ema_50 = data['Close'].ewm(span=50, adjust=False).mean()
        else:
            ema_50 = None

        if 'EMA 100' in ma_options: 
            ema_100 = data['Close'].ewm(span = 100, adjust = False).mean() 
        else: 
            ema_100 = None 
        
        return ma_50_days, ma_100_days, ma_200_days, ema_50, ema_100 

def calc_indicators(data, plot_rsi, plot_macd, plot_bollinger): 
    if plot_rsi:
        data['RSI'] = ta.rsi(data['Close'], length=14)

    if plot_macd:
        macd = ta.macd(data['Close'], fast=12, slow=26, signal=9)
        data['MACD'] = macd['MACD_12_26_9']
        data['MACD_signal'] = macd['MACDs_12_26_9']

    if plot_bollinger:
        bb_bands = ta.bbands(data['Close'], length=20, std=2)

        data['BB_upper'] = bb_bands['BBU_20_2.0']
        data['BB_middle'] = bb_bands['BBM_20_2.0']
        data['BB_lower'] = bb_bands['BBL_20_2.0']

def plot_indicators(data, plot_rsi, plot_macd, plot_bollinger): 
    if plot_rsi: 
        fig_rsi = go.Figure()
        fig_rsi.add_trace(go.Scatter(x=data.index, y=data['RSI'], mode='lines', name='RSI', line=dict(color='red')))
        fig_rsi.add_hline(y=30, line=dict(dash='dash', color='gray'), name='Oversold')
        fig_rsi.add_hline(y=70, line=dict(dash='dash', color='gray'), name='Overbought')
        fig_rsi.update_layout(
            title='Relative Strength Index (RSI)',
            xaxis_title='Date',
            yaxis_title='RSI',
            height=400,
            width=1200,
        )
        st.plotly_chart(fig_rsi)

    if plot_macd:
        fig_macd = go.Figure()
        fig_macd.add_trace(go.Scatter(x=data.index, y=data['MACD'], mode='lines', name='MACD', line=dict(color='blue')))
        fig_macd.add_trace(go.Scatter(x=data.index, y=data['MACD_signal'], mode='lines', name='MACD Signal', line=dict(color='orange')))
        fig_macd.update_layout(
            title='MACD',
            xaxis_title='Date',
            yaxis_title='Value',
            height=400,
            width=1200,
        )
        st.plotly_chart(fig_macd)

    if plot_bollinger:
        fig_bollinger = go.Figure()
        fig_bollinger.add_trace(go.Scatter(x=data.index, y=data['Close'], mode='lines', name='Stock Price', line=dict(color='green')))
        fig_bollinger.add_trace(go.Scatter(x=data.index, y=data['BB_upper'], mode='lines', name='Bollinger Upper Band', line=dict(color='red')))
        fig_bollinger.add_trace(go.Scatter(x=data.index, y=data['BB_middle'], mode='lines', name='Bollinger Middle Band', line=dict(color='blue')))
        fig_bollinger.add_trace(go.Scatter(x=data.index, y=data['BB_lower'], mode='lines', name='Bollinger Lower Band', line=dict(color='red')))
        fig_bollinger.update_layout(
            title='Bollinger Bands',
            xaxis_title='Date',
            yaxis_title='Price',
            height=600,
            width=1200,
        )
        st.plotly_chart(fig_bollinger)
