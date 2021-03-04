
import librosa
from os import listdir
import numpy as np
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.nn.functional as F
from main import RNN


model = RNN()
loss_fn = torch.nn.CrossEntropyLoss()
learning_rate = 0.0001
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
for t in range(1000):
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


torch.save(model.state_dict(), "C:/Users/Somn117/PycharmProjects/voicer/model.pt")

loadedSound=[]
X, sr = librosa.load("C:/Users/Somn117/PycharmProjects/voicer/test/1.au")
loadedSound.append(librosa.feature.mfcc(X, sr=sr))
X_out=np.array(loadedSound)
output = model(X_out).detach().numpy()
output_loss = loss_fn(y_pred, y_train_torch)
print(output)