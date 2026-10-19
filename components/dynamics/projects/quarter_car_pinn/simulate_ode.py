import numpy as np
from scipy.integrate import solve_ivp
import pandas as pd
import os
import matplotlib.pyplot as plt

def build_road_profile(t):
    """
    Creates a road profile function h(t).
    For instance, a bumpy road or an isolated speed bump.
    """
    # Simply an isolated speed bump
    h = np.zeros_like(t)
    # A bump between t=1.0s and t=2.0s
    bump_idx = (t >= 1.0) & (t <= 2.0)
    # Sinusoidal bump with amplitude 0.1m
    h[bump_idx] = 0.1 * np.sin(np.pi * (t[bump_idx] - 1.0) / 1.0)
    return h

def road_height(t):
    """Continuous road height function for the ODE solver."""
    if 1.0 <= t <= 2.0:
        return 0.1 * np.sin(np.pi * (t - 1.0) / 1.0)
    return 0.0

def road_velocity(t):
    """Derivative of road height."""
    if 1.0 <= t <= 2.0:
        return 0.1 * np.pi * np.cos(np.pi * (t - 1.0) / 1.0)
    return 0.0

def quarter_car_ode(t, state, ms, mu, ks, kt, c):
    """
    2-DOF Quarter-Car Model ODEs.
    state = [zs, zs_dot, zu, zu_dot]
    zs: displacement of sprung mass (car body)
    zu: displacement of unsprung mass (wheel)
    ms: sprung mass
    mu: unsprung mass
    ks: suspension stiffness
    kt: tire stiffness
    c: suspension damping
    """
    zs, zs_dot, zu, zu_dot = state
    
    # Road input
    zr = road_height(t)
    zr_dot = road_velocity(t)
    
    # Equations of motion
    # ms * zs_ddot = -ks*(zs - zu) - c*(zs_dot - zu_dot)
    # mu * zu_ddot = ks*(zs - zu) + c*(zs_dot - zu_dot) - kt*(zu - zr)
    
    zs_ddot = (-ks * (zs - zu) - c * (zs_dot - zu_dot)) / ms
    zu_ddot = (ks * (zs - zu) + c * (zs_dot - zu_dot) - kt * (zu - zr)) / mu
    
    return [zs_dot, zs_ddot, zu_dot, zu_ddot]

def simulate_and_save(output_dir="data", num_samples=1000):
    """
    Generate synthetic data for training the world model.
    Vary stiffness (ks) and damping (c).
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Fixed parameters
    ms = 300.0  # kg, 1/4 of total car body mass
    mu = 50.0   # kg, wheel and axle mass
    kt = 200000.0 # N/m, tire stiffness
    
    # Time vector
    t_span = (0.0, 5.0)
    t_eval = np.linspace(*t_span, 500) # 100Hz sampling
    
    all_data = []
    
    # Random uniform ranges for ks and c
    # ks: 10000 to 50000 N/m
    # c: 1000 to 5000 Ns/m
    np.random.seed(42)
    ks_samples = np.random.uniform(10000, 50000, num_samples)
    c_samples = np.random.uniform(1000, 5000, num_samples)
    
    print(f"Generating {num_samples} rollout simulations...")
    
    for i in range(num_samples):
        ks = ks_samples[i]
        c = c_samples[i]
        
        # Initial conditions: [zs, zs_dot, zu, zu_dot] = zero
        y0 = [0.0, 0.0, 0.0, 0.0]
        
        sol = solve_ivp(quarter_car_ode, t_span, y0, t_eval=t_eval, 
                        args=(ms, mu, ks, kt, c), method='RK45')
        
        # Calculate road input at these times
        zr = np.array([road_height(ti) for ti in sol.t])
        
        # Build dataframe for this rollout
        df = pd.DataFrame({
            'time': sol.t,
            'zs': sol.y[0],
            'zs_dot': sol.y[1],
            'zu': sol.y[2],
            'zu_dot': sol.y[3],
            'road': zr,
            'ks': ks,
            'c': c,
            'rollout_id': i
        })
        
        all_data.append(df)
        
    full_df = pd.concat(all_data, ignore_index=True)
    save_path = os.path.join(output_dir, "quarter_car_dataset.csv")
    full_df.to_csv(save_path, index=False)
    print(f"Data saved to {save_path}")

if __name__ == "__main__":
    simulate_and_save()
