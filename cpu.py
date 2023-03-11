import csv
import numpy as np
# Загрузка данных из файла CSV
data = []
with open('bnb.csv', 'r') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    for row in reader:
        data.append(row)
# Удаление первой строки
data.pop(0)
# Преобразование данных в numpy массив
data_np = np.array(data).astype(np.float)
# Разделение данных на признаки и целевой вектор
X = data_np[:, 1:5] # Признаки (open, high, low, close)
y = data_np[:, 5] # Целевой вектор (volume)
# Нормализация данных
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X)
# Разбиение данных на тренировочные и тестовые
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
# Создание модели нейронной сети
from keras.models import Sequential
from keras.layers import Dense
model = Sequential()
model.add(Dense(units=12, activation='relu', input_dim=4))
model.add(Dense(units=8, activation='relu'))
model.add(Dense(units=1, activation='linear'))
model.compile(loss='mse', optimizer='adam', metrics=['mae'])
# Обучение модели
model.fit(X_train, y_train, epochs=100, batch_size=32, verbose=2)
# Прогноз курса криптовалюты
y_pred = model.predict(X_test)
# Оценка точности прогноза
from sklearn.metrics import mean_squared_error
mse = mean_squared_error(y_test, y_pred)
print('Mean Squared Error: ', mse)