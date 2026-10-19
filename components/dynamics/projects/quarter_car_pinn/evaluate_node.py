import torch
import torch.nn as nn
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from simulate_ode import quarter_car_ode, road_height, road_velocity
from train_world_model_node import NeuralODEWorldModel
from train_world_model import WorldModel as PINNWorldModel
from optimize_suspension import get_road_profile

def load_node_model(model_path="models/world_model_node.pth"):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    
    model = NeuralODEWorldModel().to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    for param in model.parameters(): param.requires_grad = False
        
    return model, checkpoint['s_mean'].to(device), checkpoint['s_std'].to(device), checkpoint['ap_mean'].to(device), checkpoint['ap_std'].to(device), device

def load_pinn_model(model_path="models/world_model.pth"):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    
    model = PINNWorldModel().to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    for param in model.parameters(): param.requires_grad = False
        
    return model, checkpoint['in_mean'].to(device), checkpoint['in_std'].to(device), checkpoint['out_mean'].to(device), checkpoint['out_std'].to(device), device

def rollout_node(model, init_state, road_h, road_v, ks, c, s_mean, s_std, ap_mean, ap_std, device):
    states = [init_state] # shape (1, 4)
    dt = 0.01
    
    steps = len(road_h)
    current_state = init_state
    
    for i in range(steps - 1):
        road_input = torch.tensor([[road_h[i], road_v[i]]], dtype=torch.float32, device=device)
        params = torch.tensor([[ks, c]], dtype=torch.float32, device=device)
        
        ap = torch.cat([road_input, params], dim=1)
        ap_norm = (ap - ap_mean) / ap_std
        s_norm = (current_state - s_mean) / s_std
        
        with torch.no_grad():
            next_state_norm = model(s_norm, ap_norm)
            
        next_state = next_state_norm * s_std + s_mean
        states.append(next_state)
        current_state = next_state
        
    return torch.cat(states, dim=0).cpu().numpy()

def rollout_pinn(model, init_state, road_h, road_v, ks, c, in_mean, in_std, out_mean, out_std, device):
    states = [init_state] 
    steps = len(road_h)
    current_state = init_state
    
    for i in range(steps - 1):
        road_input = torch.tensor([[road_h[i], road_v[i]]], dtype=torch.float32, device=device)
        params = torch.tensor([[ks, c]], dtype=torch.float32, device=device)
        x = torch.cat([current_state, road_input, params], dim=1)
        x_norm = (x - in_mean) / in_std
        
        with torch.no_grad():
            next_state_norm = model(x_norm)
            
        next_state = next_state_norm * out_std + out_mean
        states.append(next_state)
        current_state = next_state
        
    return torch.cat(states, dim=0).cpu().numpy()

def evaluate():
    print("Loading models...")
    node_model, ns_mean, ns_std, nap_mean, nap_std, device = load_node_model()
    pinn_model, pin_mean, pin_std, pout_mean, pout_std, _ = load_pinn_model()
    
    # We test on an extreme condition to test OOD generalizability from the rebalancing
    # Let's use a very stiff spring and a bumpy road
    ks_test = 48000.0
    c_test = 1200.0
    
    t_span = (0.0, 5.0)
    t_eval, road_h, road_v = get_road_profile(t_span)
    
    print("Running Ground Truth ODE...")
    sol_true = solve_ivp(quarter_car_ode, t_span, [0.0, 0.0, 0.0, 0.0], t_eval=t_eval, 
                        args=(300.0, 50.0, ks_test, 200000.0, c_test), method='RK45')
    traj_true_zs = sol_true.y[0]
    
    init_state = torch.tensor([[0.0, 0.0, 0.0, 0.0]], dtype=torch.float32, device=device)
    
    print("Running Standard PINN Rollout...")
    traj_pinn = rollout_pinn(pinn_model, init_state, road_h, road_v, ks_test, c_test, pin_mean, pin_std, pout_mean, pout_std, device)
    
    print("Running Neural ODE Rollout...")
    traj_node = rollout_node(node_model, init_state, road_h, road_v, ks_test, c_test, ns_mean, ns_std, nap_mean, nap_std, device)
    
    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(t_eval, traj_true_zs, 'k-', linewidth=2, label='Ground Truth (ODE Solver)')
    plt.plot(t_eval, traj_pinn[:, 0], 'r--', linewidth=2, label='Standard PINN Surrogate')
    plt.plot(t_eval, traj_node[:, 0], 'b-.', linewidth=2, label='Neural ODE (with Rebalancing)')
    plt.title(f'Continuous Rollout on Hard Condition (k={ks_test} N/m, c={c_test} Ns/m)')
    plt.xlabel('Time (s)')
    plt.ylabel('Sprung Mass Displacement (m)')
    plt.legend()
    plt.grid(True)
    plt.savefig('results/node_vs_pinn.png')
    print("Saved comparison to results/node_vs_pinn.png")
    
if __name__ == "__main__":
    evaluate()
