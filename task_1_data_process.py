import pandas as pd
import numpy as np

def get_random_data(X, Y, train_indices):
    test_indices = np.array(list(set(range(len(X))) - set(train_indices)))
    X_train = X[train_indices]
    X_test = X[test_indices]
    Y_train = Y[train_indices]
    Y_test = Y[test_indices]
    return X_train, Y_train, X_test, Y_test

def get_data_train():
    df = pd.read_excel("train.xlsx")
    data = df.values

    N = data.shape[0]

    train_indices = np.random.choice(N, round(N*0.85), replace=False)
    test_indices = np.array(list(set(range(N)) - set(train_indices)))

    data_test =data[train_indices, :]

    Names = data_test[:, 0]
    Names = list(Names)
    Names = ','.join(Names)
    Names = Names.upper()
    Names = Names.split(',')
    NAMES = ""
    for i in range(len(Names)):
        if Names.count(Names[i])!=1:
            NAMES += Names[i]
            if i != len(Names)-1:
                NAMES+=','
    NAMES = NAMES.split(',')
    NAMES = set(NAMES)
    NAMES = list(NAMES)

    X = np.zeros((N , len(NAMES) + 5))

    for i in range(N):
        Name = df.iloc[i, 0]
        Name = Name.split(',')
        for j in range(len(Name)):
            if NAMES.count(Name[j].upper()) != 0:
                X[i][NAMES.index(Name[j].upper())] = 1
        X[i][len(NAMES)] = data[i, 1]
        X[i][len(NAMES) + 1] = data[i, 3]
        X[i][len(NAMES) + 2] = data[i, 4]
        X[i][len(NAMES) + 3] = data[i, 6]
        X[i][len(NAMES) + 4] = data[i, 9]

    Y_temp = data[:, data.shape[1]-1]
    
    Y_temp = Y_temp.astype(int)
    Y = np.zeros((N, 5))
    for i in range(N):
        Y[i][Y_temp[i] - 1] = 1
        
    return X, Y, NAMES, train_indices

def get_norm_data(NAMES, X):
    for i in range(len(NAMES)):
        m = X[:,i].mean()
        s = X[:,i].std()
        if s != 0:
            X[:,i] = (X[:,i] - m) / s
    return X

def get_data_test(NAMES):
    df = pd.read_excel("test.xlsx")
    data = df.values

    N = data.shape[0]

    X = np.zeros((N , len(NAMES)+5))

    for i in range(N):
        Name = df.iloc[i, 0]
        Name = Name.split(',')
        for j in range(len(Name)):
            if NAMES.count(Name[j].upper()) != 0:
                X[i][NAMES.index(Name[j].upper())] = 1
        X[i][len(NAMES)] = data[i, 1]
        X[i][len(NAMES) + 1] = data[i, 3]
        X[i][len(NAMES) + 2] = data[i, 4]
        X[i][len(NAMES) + 3] = data[i, 6]
        X[i][len(NAMES) + 4] = data[i, 9]
    return X
