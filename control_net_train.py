import os
#OpenMP error on Windows
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import pandas as pd
import copy
import numpy as np
from models import PoseControlNet
import yaml
import glob
from utils import plotter

def get_dataloaders(config: dict):
    path = config['parquet_path']
    if os.path.isdir(path):
        files = glob.glob(os.path.join(path, "*.parquet"))
        if not files:
            raise FileNotFoundError("parquets aren't found")
        df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    else:
        df = pd.read_parquet(path)

    y_cols = df.filter(like='target').columns
    x_cols = [col for col in df.columns if col not in y_cols]
    
    X = df[x_cols].values
    y = df[y_cols].values

    del df  # Free up memory

    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y, 
        test_size=config['test_size'], 
        random_state=config['random_seed']
    )

    # apply Standard Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_raw)
    X_test_scaled = scaler.transform(X_test_raw)

    scaler_path = config['scaler_path']
    joblib.dump(scaler, scaler_path)
    print(f"StandardScaler saved to: {scaler_path}")

    X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32)
    
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False, pin_memory=True)
    
    return train_loader, test_loader

# --- 2. TRAINING ENGINE (Logic) ---
def train_one_epoch(model, dataloader, criterion, optimizer, device, l1_lambda):
    model.train()
    running_loss = 0.0
    
    for inputs, targets in dataloader:
        inputs, targets = inputs.to(device), targets.to(device)

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        #l1_norm = sum(p.abs().sum() for p in model.parameters())
        #loss = loss + (l1_lambda * l1_norm)

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
    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG['learning_rate'], weight_decay=CONFIG['l2'])

    epochs_no_improve=0
    best_test_loss=np.inf
    best_model=model
    # Training Loop
    training_losses=[]
    val_losses=[]
    print("Starting training...")
    for epoch in range(CONFIG['num_epochs']):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device, CONFIG['l1'])
        training_losses.append(train_loss)
        val_loss = evaluate(model, test_loader, criterion, device)
        val_losses.append(val_loss)
        if val_loss > best_test_loss: #early stopping logic based on MSE
            epochs_no_improve+=1
            if epochs_no_improve > CONFIG['patience']:
                print(f"Early Stopping at Epoch: {epoch}")
                print(f"best model training stopped at epoch {epoch - CONFIG['patience']}")
                break
        else:
            epochs_no_improve=0
            best_test_loss=val_loss
            best_model=copy.deepcopy(model)
        #log train loss and val loss
        print(f"Epoch [{epoch+1}/{CONFIG['num_epochs']}] | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")

    # Final Evaluation
    final_loss = evaluate(best_model, test_loader, criterion, device)
    print(f"Final Test MSE Loss: {final_loss:.6f}")
    #print(f"model weights: {best_model.net[0].weight}")
    #plot training and val losses
    plotter(training_losses, val_losses)
    if CONFIG['save']:
        save_path=r'W:\VSCode\drone_project\weights\pose_control_model.pt'
        torch.save(best_model.state_dict(), save_path)
        

if __name__ == "__main__":
    main()