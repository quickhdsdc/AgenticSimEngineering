import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as patches
import torch
from scipy.integrate import solve_ivp
from simulate_ode import quarter_car_ode
from optimize_suspension import get_road_profile
from evaluate_node import load_node_model, load_pinn_model, rollout_node, rollout_pinn

def create_animation_3():
    print("Loading models...")
    node_model, ns_mean, ns_std, nap_mean, nap_std, device = load_node_model()
    pinn_model, pin_mean, pin_std, pout_mean, pout_std, _ = load_pinn_model()
    
    ks_test = 48000.0
    c_test = 1200.0
    ms = 300.0
    mu = 50.0
    kt = 200000.0
    
    t_span = (0.0, 5.0)
    t_eval, road_h, road_v = get_road_profile(t_span)
    
    print("Computing Ground-Truth ODE integration...")
    sol_true = solve_ivp(quarter_car_ode, t_span, [0.0, 0.0, 0.0, 0.0], t_eval=t_eval, 
                        args=(ms, mu, ks_test, kt, c_test), method='RK45')
    zs_true = sol_true.y[0]
    zu_true = sol_true.y[2]
    
    init_state = torch.tensor([[0.0, 0.0, 0.0, 0.0]], dtype=torch.float32, device=device)
    
    print("Running Standard PINN Rollout...")
    traj_pinn = rollout_pinn(pinn_model, init_state, road_h, road_v, ks_test, c_test, pin_mean, pin_std, pout_mean, pout_std, device)
    zs_pinn = traj_pinn[:, 0]
    zu_pinn = traj_pinn[:, 2]
    
    print("Running Neural ODE Rollout...")
    traj_node = rollout_node(node_model, init_state, road_h, road_v, ks_test, c_test, ns_mean, ns_std, nap_mean, nap_std, device)
    zs_node = traj_node[:, 0]
    zu_node = traj_node[:, 2]
    
    # Set up the figure
    print("Setting up animation...")
    fig, (ax_ode, ax_pinn, ax_node) = plt.subplots(1, 3, figsize=(15, 6))
    fig.suptitle(f'Quarter-Car Hard Condition (k={ks_test:.0f} N/m, c={c_test:.0f} Ns/m)\nComparing Simulation Architectures', fontsize=14)
    
    def setup_ax(ax, title):
        ax.set_title(title)
        ax.set_xlim(-0.5, 0.5)
        ax.set_ylim(-0.2, 1.2)
        ax.set_aspect('equal')
        ax.axis('off')
        
        # Ground line
        ground, = ax.plot([-0.5, 0.5], [0, 0], 'k-', lw=2)
        
        # Wheel (unsprung mass)
        wheel_radius = 0.15
        wheel = patches.Circle((0, wheel_radius), wheel_radius, fill=False, color='black', lw=2)
        ax.add_patch(wheel)
        
        # Car body (sprung mass)
        car_width = 0.6
        car_height = 0.3
        car_y_base = 0.6
        car = patches.Rectangle((-car_width/2, car_y_base), car_width, car_height, fill=True, color='royalblue', alpha=0.8)
        ax.add_patch(car)
        
        # Spring
        spring, = ax.plot([], [], 'r-', lw=2)
        # Damper
        damper_outer, = ax.plot([], [], 'b-', lw=3)
        damper_inner, = ax.plot([], [], 'b-', lw=1.5)
        # Road bump
        bump, = ax.plot([], [], 'g-', lw=2)
        
        return {'wheel': wheel, 'car': car, 'spring': spring, 
                'damper_outer': damper_outer, 'damper_inner': damper_inner, 'bump': bump,
                'wheel_r': wheel_radius, 'car_y_base': car_y_base}

    elements_ode = setup_ax(ax_ode, 'Ground-Truth ODE')
    elements_pinn = setup_ax(ax_pinn, 'Standard PINN (Drifting)')
    elements_node = setup_ax(ax_node, 'Neural ODE PINN (Stable)')
    
    time_text = fig.text(0.5, 0.05, '', ha='center', fontsize=12)

    def draw_spring(ax_elements, x, y_bottom, y_top, coils=5, width=0.1):
        n_points = coils * 4 + 1
        y_points = np.linspace(y_bottom, y_top, n_points)
        x_points = np.zeros(n_points) + x
        for i in range(1, n_points-1):
            if i % 4 == 1: x_points[i] += width/2
            elif i % 4 == 3: x_points[i] -= width/2
        ax_elements['spring'].set_data(x_points, y_points)
        
    def draw_damper(ax_elements, x, y_bottom, y_top):
        mid_y = (y_bottom + y_top) / 2
        ax_elements['damper_outer'].set_data([x, x], [y_bottom, mid_y + 0.05])
        ax_elements['damper_inner'].set_data([x, x], [mid_y, y_top])

    def update(frame):
        t = t_eval[frame]
        r_h = road_h[frame]
        time_text.set_text(f'Time: {t:.2f} s')
        
        def update_side(elems, zs, zu):
            w_r = elems['wheel_r']
            c_base_y = elems['car_y_base']
            
            wheel_y = r_h + w_r + zu
            elems['wheel'].center = (0, wheel_y)
            
            car_bottom_y = c_base_y + zs
            elems['car'].set_y(car_bottom_y)
            
            spring_x = -0.15
            damper_x = 0.15
            
            draw_spring(elems, spring_x, wheel_y, car_bottom_y)
            draw_damper(elems, damper_x, wheel_y, car_bottom_y)
            
            x_road = np.linspace(-0.5, 0.5, 100)
            y_road = np.zeros_like(x_road)
            v_x = 1.0
            x_car = t * v_x
            
            for i, x in enumerate(x_road):
                x_world = x_car + x
                if 1.0 <= x_world <= 2.0:
                    y_road[i] = 0.1 * np.sin(np.pi * (x_world - 1.0) / 1.0)
            
            elems['bump'].set_data(x_road, y_road)
            
            return [elems['wheel'], elems['car'], elems['spring'], elems['damper_outer'], elems['damper_inner'], elems['bump']]

        idx_pinn = min(frame, len(zs_pinn)-1) 
        idx_node = min(frame, len(zs_node)-1) 
        
        arts = [time_text]
        arts += update_side(elements_ode, zs_true[frame], zu_true[frame])
        arts += update_side(elements_pinn, zs_pinn[idx_pinn], zu_pinn[idx_pinn])
        arts += update_side(elements_node, zs_node[idx_node], zu_node[idx_node])
        
        return arts

    print("Rendering animation (this may take a minute to save)...")
    step = 4
    frames = range(0, len(t_eval), step)
    
    ani = animation.FuncAnimation(fig, update, frames=frames, interval=20, blit=True)
    
    save_path = 'results/quarter_car_animation_3.gif'
    try:
        writer = animation.PillowWriter(fps=25)
        ani.save(save_path, writer=writer)
        print(f"Animation saved to {save_path}")
    except Exception as e:
        print(f"Failed to generate GIF: {e}")

if __name__ == "__main__":
    create_animation_3()
