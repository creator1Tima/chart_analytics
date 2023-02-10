import numpy as np 
import pyopencl as cl 
import os 
# Загрузка данных и подготовка к обучению 
data = np.loadtxt('bnb.csv', delimiter=',') 
X = data[:, 0:4] # Входные данные 
y = data[:, 4] # Целевые данные 
# Настройка OpenCL 
os.environ['PYOPENCL_CTX'] = '0' # Установка нужного GPU-устройства 
ctx = cl.create_some_context() # Создание контекста OpenCL 
queue = cl.CommandQueue(ctx) # Создание очереди OpenCL 
mf = cl.mem_flags # Флаги памяти OpenCL 
# Преобразование X, y в OpenCL-буферы 
X_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=X) 
y_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=y)  
# Определение функции OpenCL, которую мы будем использовать 
prg = cl.Program(ctx, """   __kernel void predict(__global float *X, __global float *y, __global float *pred) {   int gid = get_global_id(0);   pred[gid] = 0;   for (int i=0; i<4; i++) {     pred[gid] += X[gid*4+i] * y[i];   } } """).build() 
# Выделяем буферы, соответствующие прогнозу 
pred_buf = cl.Buffer(ctx, mf.WRITE_ONLY, y.nbytes) 
# Запуск OpenCL-ядра с X, y-буферы  
prg.predict(queue, X.shape, None, X_buf, y_buf, pred_buf)  
# Копирование OpenCL-буфера pred_buf в numpy-массив pred 
pred = np.empty_like(y)  
cl.enqueue_copy(queue, pred, pred_buf).wait()  
# Вывод результатов 
print("Прогноз:", pred)
