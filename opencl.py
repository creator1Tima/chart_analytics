import pandas as pd
import tensorflow as tf
import numpy as np
import opencl
# Загрузка данных для обучения
data = pd.read_csv('bnb.csv')
# Преобразование данных в массивы NumPy
X_train = data.iloc[:, 1:5].values
y_train = data.iloc[:, 5].values
# Определение структуры модели
model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(4, activation='relu'))
model.add(tf.keras.layers.Dense(1))
# Компиляция модели
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
# Обучение модели
with opencl.OpenCLContext() as cl_context:
    model.fit(X_train, y_train, epochs=10, batch_size=32, use_opencl=True, opencl_context=cl_context)
# Сохранение модели
model.save('crypto_model.h5')
