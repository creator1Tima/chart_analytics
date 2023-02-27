import csv
#import matplotlib.pylab as plt
#from keras.models import Sequential #Для cuda
#from keras.layers import Dense #Для cuda
import numpy as np

n = 1749 #колличество информации в дата сете

fridge = list()
fridge_data = list()

def get_data(data):
    return fridge[fridge_data.index(data)]


with open("csv_data_files/BNB.csv", "r") as file:
    reader = csv.reader(file)
    for row in reader:
        fridge.append(row)
        fridge_data.append(row[0])

#Создадим промежуточный список для простой работы с счётчиком
listy = []
for i in range(n):
    listy.append(i)

#print(listy)
#print(len(fridge_data))

#x = np.arange(get_data(year + "-" + mounts + "-" +  day"))
# Создание датасета с признаками, которые мы будем использовать для обучения нейронной сети
X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
y = np.array([1, 2, 3, 4])


'''
#Блок обучения для cuda
#==============================================================
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Conv1D
from tensorflow.keras.layers import MaxPooling1D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Activation
import numpy as np
# Загрузка данных
#data = tf.keras.utils.get_file('name_of_file.csv', 'https://url_of_file.csv')
data = tf.keras.utils.get_file('bnb.csv', 'https://esrdata.site/index.php/s/crypt/download/bnb.csv')
# Загрузка данных из .csv файла в массив NumPy
data_np = np.genfromtxt(data, delimiter=',', skip_header=1)
# Разделение данных на входные и выходные
X = data_np[:,1:5]
y = data_np[:,5]
# Преобразование данных в формат тензора
X = tf.convert_to_tensor(X,dtype=tf.float32)
y = tf.convert_to_tensor(y,dtype=tf.float32)
# Создание модели сети
model = Sequential()
# Добавление слоя свёртки
model.add(Conv1D(filters=64, kernel_size=2, activation='relu', input_shape=X.shape[1:]))
# Добавление слоя пулинга
model.add(MaxPooling1D(pool_size=2))
# Добавление дропаут слоя
model.add(Dropout(0.2))
# Добавление плотного слоя
model.add(Flatten())
model.add(Dense(64, activation='relu'))
# Добавление активационного слоя
model.add(Activation('sigmoid'))
# Добавление выходного слоя
model.add(Dense(1))
# Компиляция модели
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
# Обучение модели
model.fit(X, y, batch_size=32, epochs=50, verbose=2)
# Сохранение модели
model.save('name_of_model.h5')
#=============================================================
'''
