# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pmdarima import auto_arima
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Data loading & preprocessing
df = pd.read_csv('AAPL_preprocessed.csv', parse_dates=['Date'], index_col='Date')
df = df.sort_index()
close_ts = df['Close']

# 检查数据范围（确保单位正确）
print("数据统计信息:\n", close_ts.describe())

# %%
# Visuallize closing price & volume
df[["Close","Volume"]].plot(subplots=True, layout=(2,1));

# %%
# Divide training and test set（the last 30 days）
train_size = -30  # 测试集为最后30天
train = close_ts[:train_size]
test = close_ts[train_size:]

# %%
# Automatically select the optimal ARIMA parameters
model = auto_arima(
    train,
    seasonal=False,        # 非季节性数据
    trace=True,             # 打印搜索过程
    error_action='ignore',  # 忽略无法收敛的模型
    suppress_warnings=True,
    stepwise=True           # 加速搜索
)
print("\n最优模型参数:", model.order)

# %%
# Model fitting（auto_arima has been auto fitted）
# 输出模型摘要
print(model.summary())

# %%
# Rolling Forecast
predictions = []
history = list(train)  # 初始训练数据

# 逐步预测未来30天
for t in range(len(test)):
    # 预测下一步
    yhat = model.predict(n_periods=1)
    
    # 修正点：直接追加预测值（无需索引）
    predictions.append(yhat)  # 修改为 predictions.append(yhat)
    
    # 将真实值添加到历史数据中（模拟实时更新）
    history.append(test.iloc[t])
    
    # 更新模型（确保传递新观测值）
    model.update([test.iloc[t]])

# %%
# Calculating metrics
mse = mean_squared_error(test, predictions)
rmse = np.sqrt(mse)
mae = mean_absolute_error(test, predictions)

print("\n评估结果:")
print(f"MSE: {mse:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"MAE: {mae:.4f}")

# %%
# Visualization
plt.figure(figsize=(12, 6))
plt.plot(train.index[-100:], train[-100:], label='Rrain')
plt.plot(test.index, test, label='True Value')
plt.plot(test.index, predictions, label='Precition', linestyle='--')
plt.title('Prediction of AAPL Closing Price')
plt.xlabel('Date')
plt.ylabel('Closing Price')
plt.legend()
plt.show()


