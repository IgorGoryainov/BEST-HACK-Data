import pandas as pd
import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam
from keras.optimizers import Adagrad
from keras.optimizers import Adadelta
from keras.constraints import max_norm
import matplotlib.pyplot as plt
from task_1_data_process import get_random_data, get_data_train, get_norm_data
from keras import backend as K

X, Y, NAMES, train_indices = get_data_train()
X = np.nan_to_num(X)

df = pd.DataFrame(NAMES,  columns=["NAMES"])
df.to_csv('NAMES1.csv' , index=False)

X = get_norm_data(NAMES, X)
X_train, Y_train, X_test, Y_test = get_random_data(X, Y, train_indices)

model = Sequential([
  Dense(1024, activation='relu', input_shape=(X_train.shape[1],), kernel_constraint=max_norm(4)),
  Dropout(0.12),
  Dense(1024, activation='relu', kernel_constraint=max_norm(3)),
  Dense(5, activation='softmax')
])

model.compile(
  optimizer='adam',
  loss='categorical_crossentropy',
  metrics=['accuracy'],
)

history = model.fit(
  X_train,
  Y_train,
  epochs=10, #975
  batch_size=100,
  validation_data=(X_test, Y_test)
)

model.save('model1.h5')
