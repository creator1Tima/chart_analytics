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
