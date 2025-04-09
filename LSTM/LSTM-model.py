# -*- coding: utf-8 -*-
"""
AAPL股价预测LSTM模型-可复现版本
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.callbacks import EarlyStopping
import tensorflow as tf

# ==================== 随机种子设置 ====================
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

# ==================== 1. 数据加载与归一化 ====================
def load_and_normalize_data(filepath):
    """加载数据并执行归一化"""
    df = pd.read_csv(filepath)
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    
    # 初始化归一化器
    scaler = MinMaxScaler()
    
    # 对全表归一化（保留原始数据副本）
    original_data = df.copy()
    normalized_data = pd.DataFrame(scaler.fit_transform(df), 
                                 columns=df.columns, 
                                 index=df.index)
    
    print("\n[归一化前统计]")
    print(original_data.describe())
    print("\n[归一化后统计]")
    print(normalized_data.describe())
    
    return normalized_data, original_data, scaler

# ==================== 2. 数据预处理 ====================
def prepare_data(normalized_df, original_df, time_steps=60):
    """创建时间序列数据集（返回带日期的测试集）"""
    features = normalized_df.drop('Close', axis=1)
    target = normalized_df[['Close']]
    
    X = features.values
    y = target.values
    
    # 创建序列数据
    X_seq, y_seq, dates = [], [], []
    for i in range(len(X) - time_steps):
        X_seq.append(X[i:i+time_steps])
        y_seq.append(y[i+time_steps, 0])
        dates.append(original_df.index[i+time_steps])  # 保存对应日期
    
    X_seq = np.array(X_seq)
    y_seq = np.array(y_seq)
    
    # 划分数据集
    train_size = int(len(X_seq) * 0.8)
    return (X_seq[:train_size], X_seq[train_size:], 
            y_seq[:train_size], y_seq[train_size:],
            dates[train_size:])  # 返回测试集日期

# ==================== 3. 模型构建 ====================
def build_model(input_shape):
    """构建LSTM模型（带随机种子初始化）"""
    model = Sequential([
        LSTM(128, return_sequences=True, 
             input_shape=input_shape,
             kernel_initializer=tf.keras.initializers.glorot_uniform(seed=SEED)),
        Dropout(0.3, seed=SEED),
        LSTM(64, kernel_initializer=tf.keras.initializers.glorot_uniform(seed=SEED)),
        Dropout(0.3, seed=SEED),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

# ==================== 4. 训练与评估 ====================
def train_and_evaluate(model, X_train, y_train, X_test, y_test, test_dates, scaler, epochs=100):
    """训练模型并保存结果"""
    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=32,
        validation_split=0.2,
        callbacks=[early_stop],
        verbose=1
    )
    
    # 绘制训练曲线
    plt.figure(figsize=(12, 5))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Normalized Loss Progression')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
    
    # 预测（归一化值）
    y_pred_norm = model.predict(X_test)
    
    # 反归一化
    close_scaler = MinMaxScaler()
    close_scaler.min_, close_scaler.scale_ = scaler.min_[scaler.feature_names_in_.tolist().index('Close')], scaler.scale_[scaler.feature_names_in_.tolist().index('Close')]
    
    y_test_orig = close_scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
    y_pred_orig = close_scaler.inverse_transform(y_pred_norm).flatten()
    
    # 计算指标
    rmse = np.sqrt(mean_squared_error(y_test_orig, y_pred_orig))
    mae = mean_absolute_error(y_test_orig, y_pred_orig)
    r2 = r2_score(y_test_orig, y_pred_orig)
    
    print("\n[反归一化后的评估结果]")
    print(f"RMSE: {rmse:.2f}")
    print(f"MAE:  {mae:.2f}")
    print(f"R²:   {r2:.4f}")
    
    # 保存结果到CSV
    save_results(test_dates, y_test_orig, y_pred_orig)
    
    return y_pred_orig, y_test_orig

def save_results(dates, true_prices, pred_prices):
    """保存测试集结果到CSV"""
    result_df = pd.DataFrame({
        'Date': dates,
        'True_Price': true_prices,
        'Predicted_Price': pred_prices
    })
    
    # 请修改为您的实际保存路径
    save_path = 'prediction_results.csv'
    result_df.to_csv(save_path, index=False)
    print(f"\n结果已保存至: {save_path}")
    print(result_df.head())

# ==================== 5. 可视化 ====================
def plot_results(y_true, y_pred):
    """绘制反归一化后的结果"""
    plt.figure(figsize=(15, 6))
    plt.plot(y_true, label='True Price', color='blue', linewidth=2)
    plt.plot(y_pred, label='Predicted', color='red', linestyle='--')
    plt.title('AAPL Stock Price Prediction')
    plt.xlabel('Time Steps')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)
    plt.show()

# ==================== 主程序 ====================
if __name__ == "__main__":
    # 参数设置
    FILE_PATH = 'AAPL_preprocessed.csv'  # 请确保此文件存在
    TIME_STEPS = 60
    EPOCHS = 50
    
    # 数据流程
    normalized_df, original_df, scaler = load_and_normalize_data(FILE_PATH)
    X_train, X_test, y_train, y_test, test_dates = prepare_data(normalized_df, original_df, TIME_STEPS)
    
    # 模型流程
    model = build_model((X_train.shape[1], X_train.shape[2]))
    print("\n[模型架构]")
    model.summary()
    
    y_pred, y_true = train_and_evaluate(model, X_train, y_train, X_test, y_test, test_dates, scaler, EPOCHS)
    plot_results(y_true, y_pred)