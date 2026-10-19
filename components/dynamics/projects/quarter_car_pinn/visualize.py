import matplotlib.pyplot as plt
import numpy as np
import os
from simulate_ode import quarter_car_ode, road_height
from scipy.integrate import solve_ivp

def plot_optimization_history():
    if not os.path.exists('results/opt_history.npy'):
        print("Optimization history not found!")
        return
        
    history = np.load('results/opt_history.npy', allow_pickle=True).item()
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    axes[0].plot(history['loss'], color='red')
    axes[0].set_title('Comfort Objective Loss')
    axes[0].set_xlabel('Epochs')
    axes[0].grid(True)
    
    axes[1].plot(history['ks'], color='blue')
    axes[1].set_title('Spring Stiffness $k$ Optimization')
    axes[1].set_xlabel('Epochs')
    axes[1].set_ylabel('N/m')
    axes[1].grid(True)
    
    axes[2].plot(history['c'], color='green')
    axes[2].set_title('Damping $c$ Optimization')
    axes[2].set_xlabel('Epochs')
    axes[2].set_ylabel('Ns/m')
    axes[2].grid(True)
    
    plt.tight_layout()
    plt.savefig('results/optimization_curves.png')
    print("Saved optimization curves to results/optimization_curves.png")

def plot_trajectories():
    if not os.path.exists('results/traj_baseline.npy'):
        print("Trajectories not found!")
        return
        
    t_eval = np.load('results/t_eval.npy')
    road_h = np.load('results/road_h.npy')
    
    # Baseline Model trajectory (ks=45000, c=1500)
    traj_baseline = np.load('results/traj_baseline.npy')
    
    # Optimized Model trajectory
    traj_opt = np.load('results/traj_opt.npy')
    
    # We also solve the true ODE for the baseline and optimal parameters to verify our World Model's accuracy
    print("Verifying against Ground-Truth ODE...")
    ks_opt = np.load('results/opt_history.npy', allow_pickle=True).item()['ks'][-1]
    c_opt = np.load('results/opt_history.npy', allow_pickle=True).item()['c'][-1]
    
    t_span = (0.0, 5.0)
    sol_baseline_true = solve_ivp(quarter_car_ode, t_span, [0.0, 0.0, 0.0, 0.0], t_eval=t_eval, 
                        args=(300.0, 50.0, 45000.0, 200000.0, 1500.0), method='RK45')
                        
    sol_opt_true = solve_ivp(quarter_car_ode, t_span, [0.0, 0.0, 0.0, 0.0], t_eval=t_eval, 
                        args=(300.0, 50.0, ks_opt, 200000.0, c_opt), method='RK45')
    
    fig, axes = plt.subplots(3, 1, figsize=(10, 10))
    
    # 1. Road Profile
    axes[0].plot(t_eval, road_h, 'k', label='Road Bump', lw=2)
    axes[0].set_title('Road Profile Input')
    axes[0].set_ylabel('Displacement (m)')
    axes[0].grid(True)
    axes[0].legend()
    
    # 2. Predicted Trajectories (Sprung Mass zs)
    axes[1].plot(t_eval, sol_baseline_true.y[0], 'b--', alpha=0.5, label='Baseline (True ODE)')
    axes[1].plot(t_eval, traj_baseline[:, 0], 'b-', label='Baseline (World Model)')
    axes[1].plot(t_eval, sol_opt_true.y[0], 'r--', alpha=0.5, label='Optimized (True ODE)')
    axes[1].plot(t_eval, traj_opt[:, 0], 'r-', label='Optimized (World Model)')
    axes[1].set_title('Sprung Mass Displacement ($z_s$)')
    axes[1].set_ylabel('Displacement (m)')
    axes[1].grid(True)
    axes[1].legend()
    
    # 3. Accelerations (Comfort)
    dt = 0.01
    
    # Baseline
    v_base = traj_baseline[:, 1]
    a_base = (v_base[1:] - v_base[:-1]) / dt
    
    # Optimized
    v_opt = traj_opt[:, 1]
    a_opt = (v_opt[1:] - v_opt[:-1]) / dt
    
    t_acc = t_eval[:-1]
    axes[2].plot(t_acc, a_base, 'b-', label='Baseline Acceleration')
    axes[2].plot(t_acc, a_opt, 'r-', label='Optimized Acceleration')
    axes[2].set_title('Passenger Comfort (Vertical Acceleration)')
    axes[2].set_xlabel('Time (s)')
    axes[2].set_ylabel('Acceleration ($m/s^2$)')
    axes[2].grid(True)
    axes[2].legend()
    
    plt.tight_layout()
    plt.savefig('results/trajectory_comparison.png')
    print("Saved trajectory comparison to results/trajectory_comparison.png")

if __name__ == "__main__":
    import matplotlib
    matplotlib.use('Agg') # non-interactive
    
    # Add a check to ensure optimization is done before trying to plot
    if not os.path.exists('results/opt_history.npy'):
        print("Waiting for optimization results to be generated...")
    else:
        plot_optimization_history()
        plot_trajectories()
