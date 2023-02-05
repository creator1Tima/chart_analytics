import csv
import matplotlib.pylab as plt
import numpy as np

n = 1749

fridge = list()
fridge_data = list()

def get_data(data):
    return fridge[fridge_data.index(data)]


with open("csv_data_files/BNB.csv", "r") as file:
    reader = csv.reader(file)
    for row in reader:
        fridge.append(row)
        fridge_data.append(row[0])

#print(len(fridge_data))
x = np.arange(get_data(year + "-" + mounts + "-" +  day"))
f = 1 / (1 + np.exp(-x))
plt.plot(x, f)
plt.xlabel('x')
plt.ylabel('f(x)')
plt.show()
