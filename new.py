import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
# Загрузка данных
df = pd.read_csv('bnb.csv')
df.head()
# Подготовка данных
# Отмасштабируем данные
scaler = MinMaxScaler(feature_range=(0, 1))
close_prices = df.loc[:, 'Close'].values.reshape(-1,1)
close_prices = scaler.fit_transform(close_prices)
# Формируем датасет
X = []
y = []
for i in range(60, len(df)):
    X.append(close_prices[i-60:i, 0])
    y.append(close_prices[i, 0])
    
X, y = np.array(X), np.array(y)
# Разделим датасет на тренировочный и тестовый
X_train = X[:int(X.shape[0]*0.8)]
X_test = X[int(X.shape[0]*0.8):]
y_train = y[:int(y.shape[0]*0.8)]
y_test = y[int(y.shape[0]*0.8):]
# Решение задачи с помощью нейронной сети
# Создадим модель
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1],1)))
model.add(LSTM(units=50))
model.add(Dense(1))
# Зададим параметры обучения
model.compile(loss='mean_squared_error', optimizer='adam')
# Обучим модель
model.fit(X_train, y_train, epochs=100, batch_size=1, verbose=2)
# Делаем прогноз
predictions = model.predict(X_test)
predictions = scaler.inverse_transform(predictions)
# Посчитаем ошибку 
rmse=np.sqrt(mean_squared_error(y_test, predictions))
print('Test RMSE: %.3f' % rmse)