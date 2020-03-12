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
from data_process import get_random_data, get_data_train, get_norm_data

X, Y, NAMES, train_indices = get_data_train()

df = pd.DataFrame(NAMES,  columns=["NAMES"])
df.to_csv('NAMES.csv' , index=False)

X = get_norm_data(NAMES, X)
X_train, Y_train, X_test, Y_test = get_random_data(X, Y, train_indices)

model = Sequential([
  Dense(1024, activation='relu', input_shape=(len(NAMES)+3,), kernel_constraint=max_norm(3)),
  Dropout(0.12),
  Dense(1024, activation='relu', kernel_constraint=max_norm(3)),
  Dense(1)
])

model.compile(
  optimizer='adam',
  loss='mean_absolute_error',
  metrics=['accuracy'],
)

history = model.fit(
  X_train,
  Y_train,
  epochs=975,
  batch_size=100,
  validation_data=(X_test, Y_test)
)

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

model.save('model.h5')
