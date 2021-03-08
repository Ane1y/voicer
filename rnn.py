import librosa
from os import listdir
import numpy as np
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.nn.functional as F
from functions import loadSound, RNN


#one = loadSound('./voice_123/one/')
#two=loadSound('./voice_123/two/')
#three=loadSound('./voice_123/three/')
#Xo=np.concatenate((one,two,three), axis=0)
me = loadSound('./test/me_text/')
shurik = loadSound('./test/shurik_text/')
anely = loadSound('./test/anely_text/')
X = np.concatenate((me,shurik,anely), axis=0)
y = np.concatenate((np.repeat(0, 10), np.repeat(1, 10), np.repeat(2, 10)), axis=0)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42, shuffle=True)
X_train = X_train.swapaxes(1,0)
X_test = X_test.swapaxes(1,0)
X_train_torch = torch.from_numpy(X_train).float()
X_test_torch = torch.from_numpy(X_test).float()
y_train_torch = torch.from_numpy(y_train).long()
y_test_torch = torch.from_numpy(y_test).long()

model = RNN()
loss_fn = torch.nn.CrossEntropyLoss()
learning_rate = 0.0001
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
for t in range(10000):
    y_pred = model(X_train_torch)
    loss = loss_fn(y_pred, y_train_torch)
    print(t, loss.item())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    with torch.no_grad():
        def accuracy_calc(X, y, type):
            correct = 0
            total_correct = 0
            outputs = model(X).detach().numpy()
            label = y.detach().numpy()
            for number in range(outputs.shape[0]):
                correct = np.argmax(outputs[number]) == label[number]
                total_correct += correct
            print(type + ' accuracy: ' + str(total_correct/outputs.shape[0] * 100) + '%')
        accuracy_calc(X_train_torch, y_train_torch, "Training")
        accuracy_calc(X_test_torch, y_test_torch, "Testing")
        torch.save(model.state_dict(), "./model.pt")

