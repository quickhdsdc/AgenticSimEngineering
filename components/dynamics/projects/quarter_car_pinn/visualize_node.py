import matplotlib.pyplot as plt
import numpy as np
import os
import torch
from simulate_ode import quarter_car_ode
from scipy.integrate import solve_ivp
from evaluate_node import load_node_model, rollout_node
from optimize_suspension import get_road_profile

def plot_trajectories_node():
    if not os.path.exists('results/opt_history.npy'):
        print("Optimization history not found!")
        return
        
    print("Loading Neural ODE model...")
    node_model, ns_mean, ns_std, nap_mean, nap_std, device = load_node_model()
    
    t_span = (0.0, 5.0)
    t_eval, road_h, road_v = get_road_profile(t_span)
    
    # Parameters
    ks_base, c_base = 45000.0, 1500.0
    history = np.load('results/opt_history.npy', allow_pickle=True).item()
    ks_opt = history['ks'][-1]
    c_opt = history['c'][-1]
    
    print("Running Ground-Truth ODE...")
    sol_baseline_true = solve_ivp(quarter_car_ode, t_span, [0.0, 0.0, 0.0, 0.0], t_eval=t_eval, 
                        args=(300.0, 50.0, ks_base, 200000.0, c_base), method='RK45')
                        
    sol_opt_true = solve_ivp(quarter_car_ode, t_span, [0.0, 0.0, 0.0, 0.0], t_eval=t_eval, 
                        args=(300.0, 50.0, ks_opt, 200000.0, c_opt), method='RK45')
                        
    init_state = torch.tensor([[0.0, 0.0, 0.0, 0.0]], dtype=torch.float32, device=device)
    
    print("Running Neural ODE Rollout for Baseline...")
    traj_baseline_node = rollout_node(node_model, init_state, road_h, road_v, ks_base, c_base, ns_mean, ns_std, nap_mean, nap_std, device)
    
    print("Running Neural ODE Rollout for Optimized...")
    traj_opt_node = rollout_node(node_model, init_state, road_h, road_v, ks_opt, c_opt, ns_mean, ns_std, nap_mean, nap_std, device)
    
    fig, axes = plt.subplots(3, 1, figsize=(10, 10))
    
    # 1. Road Profile
    axes[0].plot(t_eval, road_h, 'k', label='Road Bump', lw=2)
    axes[0].set_title('Road Profile Input')
    axes[0].set_ylabel('Displacement (m)')
    axes[0].grid(True)
    axes[0].legend()
    
    # 2. Predicted Trajectories (Sprung Mass zs)
    axes[1].plot(t_eval, sol_baseline_true.y[0], 'b--', alpha=0.5, label='Baseline (True ODE)')
    axes[1].plot(t_eval, traj_baseline_node[:, 0], 'b-', label='Baseline (Neural ODE)')
    axes[1].plot(t_eval, sol_opt_true.y[0], 'r--', alpha=0.5, label='Optimized (True ODE)')
    axes[1].plot(t_eval, traj_opt_node[:, 0], 'r-', label='Optimized (Neural ODE)')
    axes[1].set_title('Sprung Mass Displacement ($z_s$)')
    axes[1].set_ylabel('Displacement (m)')
    axes[1].grid(True)
    axes[1].legend()
    
    # 3. Accelerations (Comfort)
    dt = 0.01
    
    # Baseline
    v_base_arr = traj_baseline_node[:, 1]
    a_base = (v_base_arr[1:] - v_base_arr[:-1]) / dt
    
    # Optimized
    v_opt_arr = traj_opt_node[:, 1]
    a_opt = (v_opt_arr[1:] - v_opt_arr[:-1]) / dt
    
    t_acc = t_eval[:-1]
    axes[2].plot(t_acc, a_base, 'b-', label='Baseline Acceleration')
    axes[2].plot(t_acc, a_opt, 'r-', label='Optimized Acceleration')
    axes[2].set_title('Passenger Comfort (Vertical Acceleration)')
    axes[2].set_xlabel('Time (s)')
    axes[2].set_ylabel('Acceleration ($m/s^2$)')
    axes[2].grid(True)
    axes[2].legend()
    
    plt.tight_layout()
    plt.savefig('results/trajectory_comparison_node.png')
    print("Saved trajectory comparison to results/trajectory_comparison_node.png")

if __name__ == "__main__":
    import matplotlib
    matplotlib.use('Agg')
    plot_trajectories_node()
