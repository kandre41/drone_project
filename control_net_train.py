import os
#OpenMP error on Windows
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import pandas as pd
import copy
import numpy as np
from models import PoseControlNet
import yaml
# --- 1. DATA PREPARATION ---
def get_dataloaders(config: dict):
    df = pd.read_parquet(config['parquet_path'])
    
    y_cols = df.filter(like='target').columns
    x_cols = [col for col in df.columns if col not in y_cols]
    
    X = df[x_cols].values
    y = df[y_cols].values

    # Convert to Tensors
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32)
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X_tensor, y_tensor, 
        test_size=config['test_size'], 
        random_state=config['random_seed']
    )

    # Create DataLoaders (Mini-batching)
    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False)
    
    return train_loader, test_loader

# --- 2. TRAINING ENGINE (Logic) ---
def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    
    for inputs, targets in dataloader:
        inputs, targets = inputs.to(device), targets.to(device)

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
    return running_loss / len(dataloader)

def evaluate(model, dataloader, criterion, device):
    #validation
    model.eval()
    total_loss = 0.0
    
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item()
            
    return total_loss / len(dataloader)

# --- 3. MAIN SCRIPT ---
def main():
    with open(r'W:\VSCode\drone_project\settings\mlp_train.yaml', 'r') as file:
        CONFIG = yaml.safe_load(file)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("Loading data...")
    train_loader, test_loader = get_dataloaders(CONFIG)

    model = PoseControlNet(CONFIG['input_dim'], CONFIG['output_dim']).to(device)
    
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG['learning_rate'])

    epochs_no_improve=0
    best_test_loss=np.inf
    best_model=model
    # Training Loop
    print("Starting training...")
    for epoch in range(CONFIG['num_epochs']):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        
        val_loss = evaluate(model, test_loader, criterion, device)
        if val_loss > best_test_loss: #early stopping logic based on MSE
            epochs_no_improve+=1
            if epochs_no_improve > CONFIG['patience']:
                print(f"Early Stopping at Epoch: {epoch}")
                break
        else:
            epochs_no_improve=0
            best_test_loss=val_loss
            best_model=copy.deepcopy(model)
        #log train loss and val loss
        print(f"Epoch [{epoch+1}/{CONFIG['num_epochs']}] | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

    # Final Evaluation
    final_loss = evaluate(best_model, test_loader, criterion, device)
    print(f"Final Test Loss: {final_loss:.4f}")
    
    if CONFIG['save']:
        save_path=r'W:\VSCode\drone_project\weights\pose_control_model.pt'
        torch.save(best_model.state_dict(), save_path)

if __name__ == "__main__":
    main()