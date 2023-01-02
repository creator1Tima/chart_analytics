import csv
fridge = list()
fridge_data = list()


def get_data(data):
    return fridge[fridge_data.index(data)]


with open("csv_data_files/BNB.csv", "r") as file:
    reader = csv.reader(file)
    for row in reader:
        fridge.append(row)
        fridge_data.append(row[0])


print(get_data("2017-11-09"))
