import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import torch.nn as nn
from models import PoseControlNet
from sklearn.model_selection import train_test_split

import pandas as pd

dataset_path='W:\\VSCode\\drone_project\\datasets\\labeled_data\\demo1.parquet'

df=pd.read_parquet(path=dataset_path)
y_cols=df.filter(like='target').columns
x_cols = [col for col in df.columns if col not in y_cols]

X=df[x_cols].values
y=df[y_cols].values
X = torch.tensor(X, dtype=torch.float32)
Y = torch.tensor(y, dtype=torch.float32)
 
# Split the data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

model=PoseControlNet(13,4)
# Define the loss function and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
 
# Train the model
num_epochs = 10000
for epoch in range(num_epochs):
    # Forward pass
    outputs = model(X_train)
    loss = criterion(outputs, Y_train)
 
    # Backward and optimize
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
 
    if (epoch+1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Evaluate the model on the testing set
with torch.no_grad():
    test_outputs = model(X_test)
    test_loss = criterion(test_outputs, Y_test)
    print(f'Test Loss: {test_loss.item():.4f}')