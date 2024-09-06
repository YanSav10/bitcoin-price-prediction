# Bitcoin Price Prediction

### Description
This project predicts the price of Bitcoin based on historical data from 2017 to 2023. The dataset was uploaded to an AWS database and processed to build an LSTM (Long Short-Term Memory) neural network. With the trained model, a Flask server and user interface were created, allowing users to input a time period and receive a graph displaying the predicted Bitcoin price.

### Model Performance
The LSTM model demonstrated high accuracy in predicting Bitcoin prices:

- **Prediction Accuracy**: The model achieves an accuracy of approximately **99.4%** in predicting Bitcoin prices.
- **Error**: The Mean Absolute Error (MAE) is around **1.19%** on the training data and **1.21%** on the test data.

These metrics indicate that the model provides highly accurate predictions, though there may be slight variations due to market fluctuations.

### Installation and Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/YanSav10/bitcoin-price-prediction
   
2. Navigate to the project directory:
    ```bash
   cd server
   
3. Install the required dependencies:
    ```bash
   pip install -r requirements.txt
   
4. Run the Flask server:
    ```bash
   python main.py

5. Or use Gunicorn:
    ```bash
   gunicorn -b :5000 main:app
   
### Usage

- Open your browser and go to http://localhost:5000.
- Enter a desired time period, and the app will generate a graph showing the predicted Bitcoin price based on historical data.

### Technology Stack
- SQL (AWS)
- Python
- Deep Learning
- TensorFlow
- Flask

### Example Usage
1. Input a time range (e.g., 24 hours).
2. Receive a predicted Bitcoin price in the form of a graph.