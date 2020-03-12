mport torch
import random
import numpy as np
import pandas as pd
import csv
import math
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.backends.cudnn.deterministic = True
file = 'train.xlsx'
train = pd.read_excel(file, sheet_name='Sheet1')
X_train = np.zeros((7756, 45))
k = 0
for j in range (7756):
    for i in range(50):
        if (i > 2) and (i< 47):
            X_train[j][k] = train[train.columns[i]][j]
            k += 1
    k = 0
for i in range(7756):
    for j in range(45):
        if X_train[i][j] != X_train[i][j]:
            X_train[i][j] = 0.0
Y_train = train['Energ_Kcal'].to_numpy()
X1_train = torch.from_numpy(X_train)
X1_train = X1_train.type(torch.FloatTensor)
Y1_train = torch.from_numpy(Y_train)
Y1_train = Y1_train.type(torch.FloatTensor)
X1_train.t()
Y1_train.unsqueeze_(1)
X_test = np.zeros((1500, 45))
Y_test = np.zeros(1500)
for i in range(1500):
    val = random.randint(0, 7755)
    X_test[i] = X_train[val]
    Y_test[i] = Y_train[val]
X_test = torch.from_numpy(X_test)
X_test = X_test.float()
Y_test = torch.from_numpy(Y_test)
Y_test = Y_test.float()
X_test.t()
Y_test.unsqueeze_(1)


class Kcal_Net(torch.nn.Module):
    def __init__(self):
        super(Kcal_Net, self).__init__()
        self.fc1 = torch.nn.Linear(45, 90)
        self.act1 = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(90, 60)
        self.act2 = torch.nn.ReLU()
        self.fc3 = torch.nn.Linear(60, 30)
        self.act3 = torch.nn.ReLU()
        self.fc4 = torch.nn.Linear(30, 30)
        self.act4 = torch.nn.ReLU()
        self.fc5 = torch.nn.Linear(30, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act1(x)
        x = self.fc2(x)
        x = self.act2(x)
        x = self.fc3(x)
        x = self.act3(x)
        x = self.fc4(x)
        x = self.act4(x)
        x = self.fc5(x)
        return x
first_model = Kcal_Net()
loss = torch.nn.L1Loss()
optimizer = torch.optim.Adam(first_model.parameters(), lr = 0.0001)
batch_size = 20
for epoch in range(1000):
    order = np.random.permutation(len(X1_train))
    for start_index in range(0, len(X1_train), batch_size):
        optimizer.zero_grad()
        batch_indexes = order[start_index: start_index + batch_size]
        x_batch = X1_train[batch_indexes]
        y_batch = Y1_train[batch_indexes]
        predict = first_model.forward(x_batch)
        loss_val = loss(predict, y_batch)
        loss_val.backward()
        optimizer.step()

file1 = 'test.xlsx'
train = pd.read_excel(file1, sheet_name='Sheet1')
X_tes = np.zeros((862, 45))
k = 0
for j in range(862):
    for i in range(50):
        if (i > 1) and (i < 46):
            X_tes[j][k] = train[train.columns[i]][j]
            k += 1
    k = 0

for i in range(862):
    for j in range(45):
        if X_tes[i][j] != X_tes[i][j]:
            X_tes[i][j] = 0.0

X1_tes = torch.from_numpy(X_tes)
X1_tes = X1_tes.type(torch.FloatTensor)
X1_tes.t()
tes_preds = first_model.forward(X1_tes)

fields = ['Pred_kcal']
rows = []
for i in range(862):
    rows.append([float(tes_preds[i])])
filename = "Pred_main.csv"
with open(filename, 'w', newline='') as csvfile:
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow(fields)
    csvwriter.writerows(rows)



