import librosa
from os import listdir
import numpy as np
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.nn.functional as F
from functions import loadSound
from functions import RNN



model = RNN()
model.load_state_dict(torch.load("./model.pt"))
#me = loadSound("./test/me/")
testthree=loadSound("./test/three1/")
#sasha = loadSound("C:/Users/Somn117/PycharmProjects/voicer/test/sasha/")
#test = loadSound("C:/Users/Somn117/PycharmProjects/voicer/voice_123/two/")
X = testthree
X_torch = torch.from_numpy(X).float()
print(X_torch.size())
matrix=np.zeros(3)

output = model(X_torch).detach().numpy()
#for number in range(output.shape[0]):
#    test = np.argmax(output[number])

#    if test==0:
#        matrix[0]+=1
#    elif test==1:
#        matrix[1]+=1
#    elif test==2:
#        matrix[2]+=1
test = np.argmax(output[0])
print(test)
#print(np.argmax(matrix))

#print(output)