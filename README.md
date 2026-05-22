# DL- Developing a Recurrent Neural Network Model for Stock Prediction

## AIM
To develop a Recurrent Neural Network (RNN) model for predicting stock prices using historical closing price data.

## Problem Statement and Dataset
A Recurrent Neural Network (RNN) is a type of deep learning model designed to handle sequential data, such as time series like stock prices. It processes previous inputs through loops, allowing it to capture temporal dependencies and patterns over time. When used for stock price prediction, the RNN analyzes historical price data to learn trends and make future price estimates. Its ability to remember information across sequences makes it suitable for modeling the dynamic and seasonal nature of stock markets. Overall, RNNs help improve forecast accuracy by leveraging past data to inform future predictions.


## DESIGN STEPS
### STEP 1: 

Load and normalize data, create sequences.

### STEP 2: 
Convert data to tensors and set up DataLoader.  

### STEP 3: 

Define the RNN model architecture.

### STEP 4: 

Summarize, compile with loss and optimizer.

### STEP 5: 
Train the model with loss tracking.


### STEP 6: 

Predict on test data, plot actual vs. predicted prices.


## PROGRAM

### Name: AKSHAY KARTHICK ASR

### Register Number: 212224230015

```
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

train_df = pd.read_csv('trainset.csv')
test_df = pd.read_csv('testset.csv')

train = train_df['Close'].values.reshape(-1,1)
test = test_df['Close'].values.reshape(-1,1)

scaler = MinMaxScaler()
train_s = scaler.fit_transform(train)
test_s = scaler.transform(test)

def seq(data, w):
    x,y=[],[]
    for i in range(len(data)-w):
        x.append(data[i:i+w])
        y.append(data[i+w])
    return np.array(x), np.array(y)

w = 60
x_tr,y_tr = seq(train_s,w)
x_te,y_te = seq(test_s,w)

x_tr = torch.tensor(x_tr, dtype=torch.float32)
y_tr = torch.tensor(y_tr, dtype=torch.float32)
x_te = torch.tensor(x_te, dtype=torch.float32)
y_te = torch.tensor(y_te, dtype=torch.float32)

loader = DataLoader(TensorDataset(x_tr,y_tr), batch_size=64, shuffle=True)

class RNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.rnn = nn.RNN(1,64,2,batch_first=True)
        self.fc = nn.Linear(64,1)
    def forward(self,x):
        o,_ = self.rnn(x)
        return self.fc(o[:,-1,:])

model = RNN()

opt = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.MSELoss()

losses = []

for e in range(20):
    tot=0
    for xb,yb in loader:
        opt.zero_grad()
        out = model(xb)
        loss = loss_fn(out,yb)
        loss.backward()
        opt.step()
        tot+=loss.item()
    avg = tot/len(loader)
    losses.append(avg)
    print(f"Epoch {e+1} Loss {avg:.4f}")

plt.figure()
plt.plot(losses)
plt.title("Training Loss Over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.show()

model.eval()
with torch.no_grad():
    tr_p = model(x_tr).numpy()
    te_p = model(x_te).numpy()

tr_p = scaler.inverse_transform(tr_p)
te_p = scaler.inverse_transform(te_p)
y_tr = scaler.inverse_transform(y_tr.numpy())
y_te = scaler.inverse_transform(y_te.numpy())

plt.figure(figsize=(12,5))

x1 = range(len(y_tr))
x2 = range(len(y_tr), len(y_tr)+len(y_te))

plt.plot(x1, y_tr, label="Train Actual")
plt.plot(x1, tr_p, label="Train Pred")

plt.plot(x2, y_te, label="Test Actual")
plt.plot(x2, te_p, label="Test Pred")

plt.xlabel("Time")
plt.ylabel("Price")
plt.title("Stock Price Prediction using RNN")
plt.legend()
plt.show()

print("Predicted:", te_p[-1])
print("Actual:", y_te[-1])


```

### OUTPUT

## Training Loss Over Epochs Plot
<img width="179" height="357" alt="Screenshot 2026-05-22 at 11 00 54 AM" src="https://github.com/user-attachments/assets/9d7f51c8-577a-42f7-bc9d-a5b59e77e93f" />

<img width="632" height="446" alt="Screenshot 2026-05-22 at 11 01 06 AM" src="https://github.com/user-attachments/assets/8e77fe55-4b20-4a3a-851f-b61803091cae" />


## True Stock Price, Predicted Stock Price vs time

<img width="1026" height="468" alt="Screenshot 2026-05-22 at 11 01 30 AM" src="https://github.com/user-attachments/assets/33680714-1842-4f7a-8c36-636eeb84545a" />


### Predictions
<img width="186" height="48" alt="Screenshot 2026-05-22 at 11 01 43 AM" src="https://github.com/user-attachments/assets/58376209-4c99-4d93-93e1-189a29ba5e5f" />


## RESULT
Thus, a Recurrent Neural Network (RNN) model for predicting stock prices using historical closing price data has been developed successfully.

