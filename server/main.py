from flask import Flask, render_template, request
import requests
import pandas as pd
import tensorflow as tf
import numpy as np
import plotly.graph_objs as go
import plotly.io as pio
from sklearn.preprocessing import MinMaxScaler
app = Flask(__name__)

model = tf.keras.models.load_model('my_model.keras')

def get_realtime_data(symbol='BTCUSDT', interval='1h', limit=100):
    url = f'https://api.binance.com/api/v3/klines?symbol={symbol}&interval={interval}&limit={limit}'
    response = requests.get(url)
    data = response.json()

    df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume',
                                     'close_time', 'quote_asset_volume', 'number_of_trades',
                                     'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')

    df['month'] = df['timestamp'].dt.month
    df['weekday'] = df['timestamp'].dt.weekday
    df['year'] = df['timestamp'].dt.year
    df['day'] = df['timestamp'].dt.day
    df['hour'] = df['timestamp'].dt.hour

    df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume', 'quote_asset_volume',
             'number_of_trades', 'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume',
             'month', 'weekday', 'year', 'day', 'hour']]
    return df

def predict_future(df, periods):
    df_numeric = df.drop(columns=['timestamp'])

    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df_numeric)

    X_input = scaled_data[-60:]
    X_input = np.reshape(X_input, (1, X_input.shape[0], X_input.shape[1]))

    predictions = []
    for _ in range(periods):
        pred = model.predict(X_input)

        pred_full = np.zeros((1, 1, X_input.shape[2]))
        pred_full[0, 0, 3] = pred[0, 0]
        X_input = np.append(X_input[:, 1:, :], pred_full, axis=1)

        predictions.append(pred[0, 0])

    predictions = np.array(predictions).reshape(-1, 1)

    last_column_scaler = MinMaxScaler()
    last_column_scaler.min_, last_column_scaler.scale_ = scaler.min_[3], scaler.scale_[3]
    predicted_prices = last_column_scaler.inverse_transform(predictions)

    return predicted_prices

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        try:
            period = int(request.form.get('period', 1))
        except ValueError:
            return render_template('index.html', error="Invalid period value.")

        df = get_realtime_data()

        if df is None or len(df) < 60:
            return render_template('index.html', error="Not enough data available.")

        prediction = predict_future(df, period)

        if prediction is None:
            return render_template('index.html', error="Prediction failed.")

        historical_data = df['close'].values[-60:].astype(float)
        future_data = np.concatenate([historical_data, prediction.flatten()])

        time_series = pd.date_range(start=df['timestamp'].iloc[-60], periods=len(future_data), freq='h')

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(x=time_series[:len(historical_data)], y=historical_data, mode='lines', name='Historical Data'))
        fig.add_trace(
            go.Scatter(x=time_series[-len(prediction):], y=prediction.flatten(), mode='lines', name='Predicted Data',
                       line=dict(dash='dash')))

        fig.update_layout(
            title='BTC Price Prediction',
            xaxis_title='Time',
            yaxis_title='Price (USD)',
            yaxis=dict(range=[min(future_data), max(future_data)])
        )

        graph_html = pio.to_html(fig, full_html=False)

        return render_template('result.html', graph=graph_html)

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
