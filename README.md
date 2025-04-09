README
===========================
### COMP5152 Project Introduction
This project includes the implementation of stock closing price prediction based on Decision Tree, LSTM and ARIMA models, data preprocessing scripts and analysis reports. It is mainly used to explore the forecasting effect of different time series models on AAPL stock data.

### Environment Dependency
- Python 3.8+
- Necessary library：`numpy`, `pandas`, `matplotlib`, `scikit-learn`, `tensorflow`, `statsmodels`, `pmdarima`

### Data preprocessing
1. Raw data: `AAPL_preprocessed.csv`
2. Standardized data: `AAPL_preprocessed_Normality.csv`

### LSTM
Put LSTM-model.py and AAPL_preprocessed.csv in same directory and run the code to get the results.
````markdown
comp5152-project/
├── lstm-model.py
├── AAPL_preprocessed.csv
└── prediction_results.csv
````

### ARIMA
Put ARIMA.py and AAPL_preprocessed.csv in same directory and run the code to get the results.
````markdown
comp5152-project/
├── ARIMA.py
└── AAPL_preprocessed.csv
````
To operate the model:
>```bash
>python ARIMA.py
>```
