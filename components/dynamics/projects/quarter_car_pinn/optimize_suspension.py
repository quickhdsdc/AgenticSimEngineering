import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import copy
from train_world_model import WorldModel
from simulate_ode import road_height, road_velocity

def load_world_model(model_path="models/world_model.pth"):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    
    model = WorldModel().to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval() # Freeze world model weights
    
    for param in model.parameters():
        param.requires_grad = False
        
    return model, checkpoint['in_mean'].to(device), checkpoint['in_std'].to(device), checkpoint['out_mean'].to(device), checkpoint['out_std'].to(device), device

def get_road_profile(t_span, dt=0.01):
    steps = int((t_span[1] - t_span[0]) / dt)
    t_eval = np.linspace(*t_span, steps)
    road_h = np.array([road_height(t) for t in t_eval])
    road_v = np.array([road_velocity(t) for t in t_eval])
    return t_eval, road_h, road_v

def rollout_model(model, init_state, road_h, road_v, ks, c, in_mean, in_std, out_mean, out_std, device):
    """
    Simulate the car passing over the road using the pretrained neural network (World Model).
    Returns the predicted trajectory (differentiable wrt ks and c).
    """
    states = [init_state] # shape (1, 4)
    dt = 0.01
    
    steps = len(road_h)
    
    current_state = init_state
    
    for i in range(steps - 1):
        # inputs: [zs, zs_dot, zu, zu_dot, road, road_vel, ks, c]
        road_input = torch.tensor([[road_h[i], road_v[i]]], dtype=torch.float32, device=device)
        params = torch.stack([ks, c]).view(1, 2)
        
        x = torch.cat([current_state, road_input, params], dim=1)
        
        # normalize
        x_norm = (x - in_mean) / in_std
        
        # predict next state
        with torch.enable_grad():
            next_state_norm = model(x_norm)
            
        # unnormalize
        next_state = next_state_norm * out_std + out_mean
        
        states.append(next_state)
        current_state = next_state
        
    return torch.cat(states, dim=0)

def optimize_suspension():
    print("Loading pretrained PINN World Model...")
    model, in_mean, in_std, out_mean, out_std, device = load_world_model()
    
    # Intialize params to bad values
    # ks = 45000 (very stiff)
    # c = 1500 (under-damped)
    # We define them as trainable PyTorch tensors
    ks_param = nn.Parameter(torch.tensor([45000.0], dtype=torch.float32, device=device))
    c_param = nn.Parameter(torch.tensor([1500.0], dtype=torch.float32, device=device))
    
    # We use Adam to optimize just the suspension parameters!
    optimizer = optim.Adam([ks_param, c_param], lr=500.0) # High learning rate because parameters are large
    
    t_span = (0.0, 5.0)
    t_eval, road_h, road_v = get_road_profile(t_span)
    init_state = torch.tensor([[0.0, 0.0, 0.0, 0.0]], dtype=torch.float32, device=device)
    
    epochs = 150
    history = {'ks': [], 'c': [], 'loss': []}
    
    print(f"Starting End-to-End Optimization for Comfort...")
    print(f"Initial (ks, c): ({ks_param.item():.1f}, {c_param.item():.1f})")
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        # 1. Forward pass (Simulate using the World Model)
        traj = rollout_model(model, init_state, road_h, road_v, ks_param, c_param, in_mean, in_std, out_mean, out_std, device)
        
        # 2. Define Downstream Task Objective (Comfort Loss)
        # We want to minimize vertical accelerations of the sprung mass (zs_dot dot ≈ diff(zs_dot)/dt)
        zs_dot = traj[:, 1]
        dt = 0.01
        zs_ddot = (zs_dot[1:] - zs_dot[:-1]) / dt
        
        comfort_loss = torch.sum(zs_ddot**2)
        
        # Add penalty if they get out of expected bounds
        bounds_loss = 0.0
        if ks_param < 10000:
            bounds_loss += (10000 - ks_param)**2
        elif ks_param > 50000:
            bounds_loss += (ks_param - 50000)**2
            
        if c_param < 1000:
            bounds_loss += (1000 - c_param)**2
        elif c_param > 5000:
            bounds_loss += (c_param - 5000)**2
            
        total_loss = comfort_loss + bounds_loss
        
        # 3. Backpropagation through the physics simulation!
        total_loss.backward()
        
        optimizer.step()
        
        # Record history
        history['ks'].append(ks_param.item())
        history['c'].append(c_param.item())
        history['loss'].append(total_loss.item())
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1:03d} | Loss: {total_loss.item():.2f} | k: {ks_param.item():.1f} | c: {c_param.item():.1f}")
            
    print(f"\nOptimization Complete!")
    print(f"Final Optimized parameters: ks = {ks_param.item():.1f}, c = {c_param.item():.1f}")
    
    # Save optimized parameters and history
    os.makedirs("results", exist_ok=True)
    np.save('results/opt_history.npy', history)
    
    # Save the original baseline trajectory and the optimized trajectory
    with torch.no_grad():
        ks_baseline = torch.tensor([45000.0], dtype=torch.float32, device=device)
        c_baseline = torch.tensor([1500.0], dtype=torch.float32, device=device)
        traj_baseline = rollout_model(model, init_state, road_h, road_v, ks_baseline, c_baseline, in_mean, in_std, out_mean, out_std, device)
        traj_opt = rollout_model(model, init_state, road_h, road_v, ks_param, c_param, in_mean, in_std, out_mean, out_std, device)
        
        np.save('results/traj_baseline.npy', traj_baseline.cpu().numpy())
        np.save('results/traj_opt.npy', traj_opt.cpu().numpy())
        np.save('results/t_eval.npy', t_eval)
        np.save('results/road_h.npy', road_h)

if __name__ == "__main__":
    optimize_suspension()
