import librosa
from os import listdir
import numpy as np
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.nn.functional as F


def loadSound(path):
    soundList = listdir(path)
    loadedSound = []
    for sound in soundList:
        Y, sr = librosa.load(path + sound)
        loadedSound.append(librosa.feature.mfcc(Y, sr=sr))
    return np.array(loadedSound)


class RNN(nn.Module):
    def __init__(self):
        super(RNN, self).__init__()
        self.lstm1 = nn.LSTM(input_size=87, hidden_size=256)
        self.lstm2 = nn.LSTM(input_size=256, hidden_size=128)
        self.lstm3 = nn.LSTM(input_size=128, hidden_size=64)
        self.lstm4 = nn.LSTM(input_size=64, hidden_size=32)
        self.fc1 = nn.Linear(in_features=32, out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=64)
        self.fc3 = nn.Linear(in_features=64, out_features=32)
        self.fc4 = nn.Linear(in_features=32, out_features=3)

    def forward(self, x):
        x = torch.tanh(self.lstm1(x)[0])
        x = torch.tanh(self.lstm2(x)[0])
        x = torch.tanh(self.lstm3(x)[0])
        x = torch.tanh(self.lstm4(x)[0][0])
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x


model = RNN()
model.load_state_dict(torch.load("./model.pt"))

#onex,sr=librosa.load("C:/Users/Somn117/PycharmProjects/voicer/test/me/1.au")

#print(onex.shape[0]/sr)


#onex,sr=librosa.load("C:/Users/Somn117/PycharmProjects/voicer/test/sasha/1.au")

#print(onex.shape[0]/sr)

me = loadSound("./test/me/")
#sasha = loadSound("C:/Users/Somn117/PycharmProjects/voicer/test/sasha/")
#test = loadSound("C:/Users/Somn117/PycharmProjects/voicer/voice_123/two/")
X = me
X_torch = torch.from_numpy(X).float()
print(X_torch.size())
matrix=np.zeros(3)

output = model(X_torch).detach().numpy()
for number in range(output.shape[0]):
    test = np.argmax(output[number])
    if test==0:
        matrix[0]+=1
    elif test==1:
        matrix[1]+=1
    elif test==2:
        matrix[2]+=1

print(np.argmax(matrix))

print(output)