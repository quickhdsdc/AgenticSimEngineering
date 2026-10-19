import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from tqdm import tqdm

class QuarterCarDataset(Dataset):
    def __init__(self, csv_path):
        df = pd.read_csv(csv_path)
        
        # We need pairs of (state_t, action_t) -> state_{t+1}
        # To do this efficiently, we can group by rollout_id
        
        self.inputs = []
        self.targets = []
        
        print("Processing dataset sequences...")
        for _, group in tqdm(df.groupby('rollout_id')):
            # States: zs, zs_dot, zu, zu_dot
            states = group[['zs', 'zs_dot', 'zu', 'zu_dot']].values
            roads = group['road'].values
            road_vels = np.zeros_like(roads)
            road_vels[1:] = (roads[1:] - roads[:-1]) / 0.01  # dt = 0.01s approx
            
            ks = group['ks'].values
            c = group['c'].values
            
            # Predict next state given current state, actions, and parameters
            for i in range(len(states) - 1):
                # Input: [zs, zs_dot, zu, zu_dot, road, road_vel, ks, c]
                input_vec = np.concatenate([
                    states[i], 
                    [roads[i], road_vels[i], ks[i], c[i]]
                ])
                
                # Target: [zs_next, zs_dot_next, zu_next, zu_dot_next]
                target_vec = states[i+1]
                
                self.inputs.append(input_vec)
                self.targets.append(target_vec)
                
        self.inputs = torch.tensor(np.array(self.inputs), dtype=torch.float32)
        self.targets = torch.tensor(np.array(self.targets), dtype=torch.float32)
        
        # Normalization
        self.in_mean = self.inputs.mean(dim=0)
        self.in_std = self.inputs.std(dim=0) + 1e-8
        
        self.out_mean = self.targets.mean(dim=0)
        self.out_std = self.targets.std(dim=0) + 1e-8
        
    def __len__(self):
        return len(self.inputs)
        
    def __getitem__(self, idx):
        x = (self.inputs[idx] - self.in_mean) / self.in_std
        y = (self.targets[idx] - self.out_mean) / self.out_std
        return x, y, self.inputs[idx]

class WorldModel(nn.Module):
    def __init__(self, input_dim=8, hidden_dim=64, output_dim=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim)
        )
        
    def forward(self, x):
        return self.net(x)

def pinn_loss(pred_norm, target_norm, raw_inputs, dataset, model):
    """
    Combines Data loss (MSE) and a Physics-Informed residual.
    """
    # 1. Data MSE Loss
    mse_loss = nn.MSELoss()(pred_norm, target_norm)
    
    # 2. Physics Residual (Optional but helps generalization)
    # We unnormalize predictions to check physical consistency: 
    # e.g., zs_{t+1} ≈ zs_t + zs_dot_t * dt
    dt = 0.01
    pred_raw = pred_norm * dataset.out_std.to(pred_norm.device) + dataset.out_mean.to(pred_norm.device)
    
    zs_t = raw_inputs[:, 0]
    zs_dot_t = raw_inputs[:, 1]
    zu_t = raw_inputs[:, 2]
    zu_dot_t = raw_inputs[:, 3]
    
    zs_next_pred = pred_raw[:, 0]
    zu_next_pred = pred_raw[:, 2]
    
    # Kinematic constraints
    kinematic_loss_s = nn.MSELoss()(zs_next_pred, zs_t + zs_dot_t * dt)
    kinematic_loss_u = nn.MSELoss()(zu_next_pred, zu_t + zu_dot_t * dt)
    
    physics_loss = kinematic_loss_s + kinematic_loss_u
    
    # Total loss
    return mse_loss + 0.1 * physics_loss

def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    data_path = "data/quarter_car_dataset.csv"
    if not os.path.exists(data_path):
        print("Data not found. Please run simulate_ode.py first.")
        return
        
    dataset = QuarterCarDataset(data_path)
    dataloader = DataLoader(dataset, batch_size=4096, shuffle=True, pin_memory=True)
    
    model = WorldModel().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    epochs = 20
    
    dataset.out_mean = dataset.out_mean.to(device)
    dataset.out_std = dataset.out_std.to(device)
    
    print("Beginning Training...")
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch_x, batch_y, batch_raw in tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}"):
            batch_x, batch_y, batch_raw = batch_x.to(device), batch_y.to(device), batch_raw.to(device)
            
            optimizer.zero_grad()
            preds = model(batch_x)
            
            loss = pinn_loss(preds, batch_y, batch_raw, dataset, model)
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        print(f"Epoch {epoch+1} Loss: {total_loss / len(dataloader):.6f}")
        
    # Save the model and normalization params
    os.makedirs("models", exist_ok=True)
    torch.save({
        'model_state_dict': model.state_dict(),
        'in_mean': dataset.in_mean,
        'in_std': dataset.in_std,
        'out_mean': dataset.out_mean,
        'out_std': dataset.out_std
    }, "models/world_model.pth")
    print("World Model Saved to models/world_model.pth")

if __name__ == "__main__":
    train()
