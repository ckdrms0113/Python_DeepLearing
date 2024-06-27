import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# 주식 데이터 다운로드 및 전처리 함수
def load_data(stock_name, start_date, end_date):
    df = yf.download(stock_name, start=start_date, end=end_date)
    df = df[['Close']]
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df)
    return scaled_data, scaler

# 데이터셋 생성 함수
def create_dataset(data, time_step):
    X, Y = [], []
    for i in range(len(data)-time_step-1):
        X.append(data[i:(i+time_step), 0])
        Y.append(data[i + time_step, 0])
    return np.array(X), np.array(Y)

# 데이터 로드 및 전처리
stock_name = 'AAPL'
start_date = '2000-01-01'
end_date = '2023-01-01'
data, scaler = load_data(stock_name, start_date, end_date)

# 데이터셋 생성
time_step = 60  # 입력 시퀀스 길이
X, y = create_dataset(data, time_step)
X = X.reshape(X.shape[0], X.shape[1], 1)

# 데이터셋 분할 (학습용과 테스트용)
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# GRU 모델 정의
model = Sequential()
model.add(GRU(128, input_shape=(time_step, 1)))
model.add(Dropout(0.2))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')

# 모델 학습
model.fit(X_train, y_train, epochs=10000, batch_size=32, validation_data=(X_test, y_test), verbose=1)

# 예측 수행
train_predict = model.predict(X_train)
test_predict = model.predict(X_test)

# 예측 값 역정규화
train_predict = scaler.inverse_transform(train_predict)
y_train = scaler.inverse_transform([y_train])
test_predict = scaler.inverse_transform(test_predict)
y_test = scaler.inverse_transform([y_test])

# RMSE 계산
train_rmse = np.sqrt(mean_squared_error(y_train[0], train_predict[:,0]))
test_rmse = np.sqrt(mean_squared_error(y_test[0], test_predict[:,0]))

print(f'Train RMSE: {train_rmse}')
print(f'Test RMSE: {test_rmse}')

# 예측 결과 시각화
plt.figure(figsize=(14, 7))
plt.plot(np.arange(len(y_train[0])), y_train[0], label='True Train')
plt.plot(np.arange(len(y_train[0])), train_predict[:,0], label='Predicted Train')
plt.plot(np.arange(len(y_train[0]), len(y_train[0])+len(y_test[0])), y_test[0], label='True Test')
plt.plot(np.arange(len(y_train[0]), len(y_train[0])+len(y_test[0])), test_predict[:,0], label='Predicted Test')
plt.title('GRU Model - Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Stock Price')
plt.legend()
plt.show()
