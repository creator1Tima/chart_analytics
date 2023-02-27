import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.cluster import KMeans
# Загрузка данных
data = pd.read_csv('bnb.csv')
# Нормализация данных
scaler = MinMaxScaler()
data = scaler.fit_transform(data)
# Разделение данных на обучающие и тестовые
X_train, X_test, y_train, y_test = train_test_split(data[:,0:-2], data[:,-1], test_size=0.2, random_state=42)
# Кластеризация данных
kmeans = KMeans(n_clusters=2)
kmeans.fit(X_train)
# Построение нейросети
clf = MLPRegressor(hidden_layer_sizes=(15,15,15,15), activation='relu', solver='adam',
                   learning_rate='adaptive', max_iter=1000, learning_rate_init=0.001,
                   opencl_profile='AMD', opencl_platform_id=0, opencl_device_id=1)
# Обучение модели
clf.fit(X_train, y_train)
# Предсказание результатов
predictions = clf.predict(X_test)
# Оценка модели
mse = mean_squared_error(y_test, predictions)
print("Среднеквадратичная ошибка: ", mse)