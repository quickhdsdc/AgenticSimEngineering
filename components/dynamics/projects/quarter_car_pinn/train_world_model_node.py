import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
import pandas as pd
import numpy as np
from tqdm import tqdm
from torchdiffeq import odeint

class QuarterCarDatasetNODE(Dataset):
    def __init__(self, csv_path):
        df = pd.read_csv(csv_path)
        
        self.states = []
        self.actions_params = []
        self.targets = []
        
        print("Processing dataset sequences for Neural ODE...")
        for _, group in tqdm(df.groupby('rollout_id')):
            states = group[['zs', 'zs_dot', 'zu', 'zu_dot']].values
            roads = group['road'].values
            road_vels = np.zeros_like(roads)
            road_vels[1:] = (roads[1:] - roads[:-1]) / 0.01
            
            ks = group['ks'].values
            c = group['c'].values
            
            for i in range(len(states) - 1):
                st = states[i]
                ap = np.array([roads[i], road_vels[i], ks[i], c[i]])
                target = states[i+1]
                
                self.states.append(st)
                self.actions_params.append(ap)
                self.targets.append(target)
                
        self.states = torch.tensor(np.array(self.states), dtype=torch.float32)
        self.actions_params = torch.tensor(np.array(self.actions_params), dtype=torch.float32)
        self.targets = torch.tensor(np.array(self.targets), dtype=torch.float32)
        
        self.s_mean = self.states.mean(dim=0)
        self.s_std = self.states.std(dim=0) + 1e-8
        
        self.ap_mean = self.actions_params.mean(dim=0)
        self.ap_std = self.actions_params.std(dim=0) + 1e-8
        
    def __len__(self):
        return len(self.states)
        
    def __getitem__(self, idx):
        s_norm = (self.states[idx] - self.s_mean) / self.s_std
        ap_norm = (self.actions_params[idx] - self.ap_mean) / self.ap_std
        t_norm = (self.targets[idx] - self.s_mean) / self.s_std
        return s_norm, ap_norm, t_norm, self.states[idx], self.targets[idx]

class NODEFunc(nn.Module):
    def __init__(self, state_dim=4, action_dim=4, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, state_dim)
        )
        self.current_action_params = None
        
    def forward(self, t, state):
        # We concatenate external action & parameters to the state
        x = torch.cat([state, self.current_action_params], dim=1)
        z_dot = self.net(x)
        return z_dot

class NeuralODEWorldModel(nn.Module):
    def __init__(self, state_dim=4, action_dim=4, hidden_dim=64):
        super().__init__()
        self.odefunc = NODEFunc(state_dim, action_dim, hidden_dim)
        
    def forward(self, state, action_params):
        self.odefunc.current_action_params = action_params
        # Integrate over normalized internal delta_t using 4th order Runge Kutta
        t_span = torch.tensor([0.0, 1.0], device=state.device) 
        pred_states = odeint(self.odefunc, state, t_span, method='rk4')
        return pred_states[1]

def rebalance_data(model, states, actions_params, targets, raw_states, raw_targets, device, alpha=0.5, allocation_rate=[1, 3], batch_size=4096):
    """
    Data Rebalancing for OOD Generalization (adapted from MoSim architecture).
    Finds high-error samples (e.g., extreme bumps) and oversamples them.
    """
    print("\n--- Running Dynamic Data Rebalancing ---")
    model.eval()
    criterion = nn.MSELoss(reduction='none')
    
    dataset = TensorDataset(states, actions_params, targets)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    all_losses = []
    with torch.no_grad():
        for s, ap, tgt in tqdm(dataloader, desc="Evaluating sample difficulty"):
            s, ap, tgt = s.to(device), ap.to(device), tgt.to(device)
            preds = model(s, ap)
            losses = torch.mean(criterion(preds, tgt), dim=1).cpu()
            all_losses.append(losses)
            
    all_losses = torch.cat(all_losses, dim=0)
    
    sorted_indices = torch.argsort(all_losses)
    cut_idx = int(alpha * states.shape[0])
    
    def repeat_or_sample(tensor, rate):
        if rate >= 1:
            return tensor.repeat(int(rate), *(1 for _ in range(tensor.dim() - 1)))
        else:
            num_samples = int(len(tensor) * rate)
            indices = torch.randperm(len(tensor))[:num_samples]
            return tensor[indices]
            
    def rebalance(tensor):
        tensor_sorted = tensor[sorted_indices]
        head = tensor_sorted[:cut_idx] # Easy data
        tail = tensor_sorted[cut_idx:] # Hard data
        return torch.cat((
            repeat_or_sample(head, allocation_rate[0]),
            repeat_or_sample(tail, allocation_rate[1])
        ), dim=0)

    print(f"Original Dataset Size: {len(states)}. Oversampling worst {(1-alpha)*100:.0f}% by factor of {allocation_rate[1]}x.")
    rb_states = rebalance(states)
    rb_actions = rebalance(actions_params)
    rb_targets = rebalance(targets)
    rb_raw_s = rebalance(raw_states)
    rb_raw_t = rebalance(raw_targets)
    print(f"New Rebalanced Dataset Size: {len(rb_states)}.")
    print("----------------------------------------\n")
    
    model.train()
    return rb_states, rb_actions, rb_targets, rb_raw_s, rb_raw_t

def pinn_node_loss(pred_norm, target_norm, raw_state, raw_target, s_std, s_mean):
    # 1. Data MSE Loss on Neural ODE output
    mse_loss = nn.MSELoss()(pred_norm, target_norm)
    
    # 2. Physics Residual (kinematic coherence)
    dt = 0.01
    pred_raw = pred_norm * s_std + s_mean
    
    zs_t = raw_state[:, 0]
    zs_dot_t = raw_state[:, 1]
    zu_t = raw_state[:, 2]
    zu_dot_t = raw_state[:, 3]
    
    zs_next_pred = pred_raw[:, 0]
    zu_next_pred = pred_raw[:, 2]
    
    kinematic_loss_s = nn.MSELoss()(zs_next_pred, zs_t + zs_dot_t * dt)
    kinematic_loss_u = nn.MSELoss()(zu_next_pred, zu_t + zu_dot_t * dt)
    
    physics_loss = kinematic_loss_s + kinematic_loss_u
    
    return mse_loss + 0.1 * physics_loss

def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    data_path = "data/quarter_car_dataset.csv"
    if not os.path.exists(data_path):
        print("Data not found. Please run simulate_ode.py first.")
        return
        
    dataset = QuarterCarDatasetNODE(data_path)
    
    # Pre-extract all normalized values to pass to rebalance_data
    all_s, all_ap, all_t, all_raw_s, all_raw_t = [], [], [], [], []
    for s_n, ap_n, t_n, s, t in dataset:
        all_s.append(s_n)
        all_ap.append(ap_n)
        all_t.append(t_n)
        all_raw_s.append(s)
        all_raw_t.append(t)
        
    all_s = torch.stack(all_s)
    all_ap = torch.stack(all_ap)
    all_t = torch.stack(all_t)
    all_raw_s = torch.stack(all_raw_s)
    all_raw_t = torch.stack(all_raw_t)
    
    model = NeuralODEWorldModel().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    s_std = dataset.s_std.to(device)
    s_mean = dataset.s_mean.to(device)
    
    epochs = 20
    rebalance_epoch = 10 # We rebalance the data halfway through training once model knows what is "hard"
    
    current_s, current_ap, current_t = all_s, all_ap, all_t
    current_raw_s, current_raw_t = all_raw_s, all_raw_t
    
    def create_dataloader(s, ap, t, r_s, r_t):
        tensordata = TensorDataset(s, ap, t, r_s, r_t)
        return DataLoader(tensordata, batch_size=4096, shuffle=True, pin_memory=True if torch.cuda.is_available() else False)
        
    dataloader = create_dataloader(current_s, current_ap, current_t, current_raw_s, current_raw_t)
    
    print("Beginning Neural ODE Training...")
    for epoch in range(epochs):
        
        # Apply Data Rebalancing logic at a specific epoch (e.g. 5) like MoSim OOD mechanism
        if epoch == rebalance_epoch:
            current_s, current_ap, current_t, current_raw_s, current_raw_t = rebalance_data(
                model, all_s, all_ap, all_t, all_raw_s, all_raw_t, device, alpha=0.5, allocation_rate=[1, 3], batch_size=4096
            )
            dataloader = create_dataloader(current_s, current_ap, current_t, current_raw_s, current_raw_t)
        
        model.train()
        total_loss = 0
        for batch_s, batch_ap, batch_t, batch_raw_s, batch_raw_t in tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}"):
            batch_s, batch_ap, batch_t = batch_s.to(device), batch_ap.to(device), batch_t.to(device)
            batch_raw_s, batch_raw_t = batch_raw_s.to(device), batch_raw_t.to(device)
            
            optimizer.zero_grad()
            preds = model(batch_s, batch_ap)
            
            loss = pinn_node_loss(preds, batch_t, batch_raw_s, batch_raw_t, s_std, s_mean)
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        print(f"Epoch {epoch+1} Loss: {total_loss / len(dataloader):.6f}")
        
    os.makedirs("models", exist_ok=True)
    torch.save({
        'model_state_dict': model.state_dict(),
        's_mean': dataset.s_mean,
        's_std': dataset.s_std,
        'ap_mean': dataset.ap_mean,
        'ap_std': dataset.ap_std
    }, "models/world_model_node.pth")
    print("Neural ODE World Model successfully saved to models/world_model_node.pth")

if __name__ == "__main__":
    train()
