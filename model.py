#installing the libraries
!pip install torch torchaudio torchvision librosa numpy matplotlib
!pip install -U scikit-learn
!pip install torch


import zipfile

# Path to the zip file
zip_path = '/content/audio'

# Destination directory
extract_path = '/destination/'

# Extracting all files
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
  zip_ref.extractall(extract_path)

#Processing the data
import torch
import numpy as np
import torchaudio
import matplotlib.pyplot as plt
import librosa
from sklearn.model_selection import train_test_split
audio=[]
output=[]
i=0
while i<1000:
  audio_data, sample_rate = librosa.load("/destination/audio/for-original/training/real/file"+str(i)+".wav", sr=None)
  mel_spectrogram = librosa.feature.melspectrogram(y=audio_data, sr=sample_rate)
  mel_spectrogram=np.array(mel_spectrogram)
  mel_spectrogram=np.resize(mel_spectrogram,(128))
  audio.append(mel_spectrogram)
  output.append(1)
  i=i+1
i=0
while i<1000:
  audio_data, sample_rate = librosa.load("/destination/audio/for-original/training/fake/file"+str(i)+".wav", sr=None)
  mel_spectrogram = librosa.feature.melspectrogram(y=audio_data, sr=sample_rate)
  mel_spectrogram=np.array(mel_spectrogram)
  mel_spectrogram=np.resize(mel_spectrogram,(128))
  audio.append(mel_spectrogram)
  output.append(0)
  i=i+1
output=np.array(output)
audio=np.array(audio)
X_train, X_test, y_train, y_test = train_test_split(audio, output, test_size=0.2, random_state=42)

#creating the model
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.fc1 = nn.Linear(64 * 6 * 6, 128)  # Adjust input size based on your image dimensions
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, kernel_size=2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, kernel_size=2)
        x = torch.flatten(x, 1)

        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

#trainng the model
optimizer = torch.optim.SGD(params=model_0.parameters(), 
                            lr=0.1)
loss_fn = nn.BCEWithLogitsLoss()
epochs = 10000
for epoch in range(epochs):
    ### Training
    model_0.train()

    y_logits = model_0(torch.tensor(X_train)).squeeze() # squeeze to remove extra `1` dimensions, this won't work unless model and data are on same device
    y_pred = torch.sigmoid(y_logits) # turn logits -> pred probs -> pred labls
    y_train = torch.tensor(y_train, dtype=torch.float32)  
    loss= loss_fn(y_pred, y_train)


    optimizer.zero_grad()


    loss.backward()


    optimizer.step()

    model_0.eval()
    with torch.inference_mode():
        # 1. Forward pass
        test_logits = model_0(torch.tensor(X_test)).squeeze() 
        test_pred = torch.round(torch.sigmoid(test_logits))
        # 2. Caculate loss/accuracy
        y_test = torch.tensor(y_test, dtype=torch.float32) 
        test_loss = loss_fn(test_logits,
                            y_test)
        test_acc = accuracy_fn(y_true=y_test,
                               y_pred=test_pred)
    if epoch % 10 == 0:
        acc = accuracy_fn(y_true=y_train, y_pred=torch.round(y_pred))
        print(f"Epoch: {epoch} | Loss: {loss:.5f}, Accuracy: {acc:.2f}% | Test loss: {test_loss:.5f}, Test acc: {test_acc:.2f}%")
