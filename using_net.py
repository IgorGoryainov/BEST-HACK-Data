import pandas as pd
import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.constraints import max_norm
from keras.models import load_model
from data_process import get_data_train, get_data_test, get_norm_data

df = pd.read_csv("NAMES.csv")
NAMES = df.values
NAMES = list(NAMES)
X = get_data_test(NAMES)
X = np.array(X)
X = get_norm_data(NAMES, X)

model = load_model('model.h5')

predictions = model.predict(X)

df = pd.DataFrame(predictions,  columns=["Pred_kcal"])
df.to_csv('Pred_extra_2.csv' , index=False)

