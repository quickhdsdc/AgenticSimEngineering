import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as patches
from scipy.integrate import solve_ivp
from simulate_ode import quarter_car_ode, road_height
import os

def create_animation():
    if not os.path.exists('results/opt_history.npy') or not os.path.exists('results/traj_opt.npy'):
        print("Please run optimize_suspension.py first to generate results.")
        return

    # 1. Load data
    print("Loading data for animation...")
    t_eval = np.load('results/t_eval.npy')
    road_h = np.load('results/road_h.npy')
    
    # Load optimized parameters
    history = np.load('results/opt_history.npy', allow_pickle=True).item()
    ks_opt = history['ks'][-1]
    c_opt = history['c'][-1]
    
    # Load PINN World Model results (Optimized)
    traj_pinn = np.load('results/traj_opt.npy')
    zs_pinn = traj_pinn[:, 0]
    zu_pinn = traj_pinn[:, 2]
    
    print("Computing Ground-Truth ODE integration...")
    # Compute True ODE results (Optimized parameters)
    ms = 300.0
    mu = 50.0
    kt = 200000.0
    t_span = (t_eval[0], t_eval[-1])
    y0 = [0.0, 0.0, 0.0, 0.0]
    
    sol_true = solve_ivp(quarter_car_ode, t_span, y0, t_eval=t_eval, 
                        args=(ms, mu, ks_opt, kt, c_opt), method='RK45')
                        
    zs_true = sol_true.y[0]
    zu_true = sol_true.y[2]
    
    # 2. Set up the figure
    print("Setting up animation...")
    fig, (ax_ode, ax_pinn) = plt.subplots(1, 2, figsize=(12, 6))
    fig.suptitle(f'Quarter-Car Simulation\nOptimized Params: k={ks_opt:.1f} N/m, c={c_opt:.1f} Ns/m', fontsize=14)
    
    def setup_ax(ax, title):
        ax.set_title(title)
        ax.set_xlim(-0.5, 0.5)
        # Assuming maximum displacement won't exceed these bounds drastically
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
        car_y_base = 0.6 # Initial resting height of the bottom of the car
        car = patches.Rectangle((-car_width/2, car_y_base), car_width, car_height, fill=True, color='royalblue', alpha=0.8)
        ax.add_patch(car)
        
        # Spring representation (a zig-zag line)
        spring, = ax.plot([], [], 'r-', lw=2)
        
        # Damper representation (a simple line or two parallel lines)
        damper_outer, = ax.plot([], [], 'b-', lw=3)
        damper_inner, = ax.plot([], [], 'b-', lw=1.5)
        
        # Road bump
        bump, = ax.plot([], [], 'g-', lw=2)
        
        return {'wheel': wheel, 'car': car, 'spring': spring, 
                'damper_outer': damper_outer, 'damper_inner': damper_inner, 'bump': bump,
                'wheel_r': wheel_radius, 'car_y_base': car_y_base}

    elements_ode = setup_ax(ax_ode, 'Ground-Truth ODE')
    elements_pinn = setup_ax(ax_pinn, 'PINN World Model')
    
    # Time text
    time_text = fig.text(0.5, 0.05, '', ha='center', fontsize=12)

    def draw_spring(ax_elements, x, y_bottom, y_top, coils=5, width=0.1):
        # Create zig-zag points for spring
        n_points = coils * 4 + 1
        y_points = np.linspace(y_bottom, y_top, n_points)
        x_points = np.zeros(n_points) + x
        for i in range(1, n_points-1):
            if i % 4 == 1:
                x_points[i] += width/2
            elif i % 4 == 3:
                x_points[i] -= width/2
        ax_elements['spring'].set_data(x_points, y_points)
        
    def draw_damper(ax_elements, x, y_bottom, y_top):
        mid_y = (y_bottom + y_top) / 2
        # Outer cylinder (attached to bottom)
        ax_elements['damper_outer'].set_data([x, x], [y_bottom, mid_y + 0.05])
        # Inner piston (attached to top)
        ax_elements['damper_inner'].set_data([x, x], [mid_y, y_top])

    # 3. Animation update function
    def update(frame):
        # Current time and road height
        t = t_eval[frame]
        r_h = road_h[frame]
        
        time_text.set_text(f'Time: {t:.2f} s')
        
        # Function to update one side
        def update_side(elems, zs, zu):
            w_r = elems['wheel_r']
            c_base_y = elems['car_y_base']
            
            # Update wheel position (road height + natural radius + displacement)
            wheel_y = r_h + w_r + zu
            elems['wheel'].center = (0, wheel_y)
            
            # Update car body position (natural resting base + displacement)
            car_bottom_y = c_base_y + zs
            elems['car'].set_y(car_bottom_y)
            
            # Draw spring and damper between wheel center and car bottom
            # Spring on the left, damper on the right
            spring_x = -0.15
            damper_x = 0.15
            
            draw_spring(elems, spring_x, wheel_y, car_bottom_y)
            draw_damper(elems, damper_x, wheel_y, car_bottom_y)
            
            # Draw road bump passing under (simulated by moving the bump backwards)
            # We'll just draw the local bump shape centered at 0 if the car is currently over it
            # The bump is between t=1.0 and t=2.0
            x_road = np.linspace(-0.5, 0.5, 100)
            y_road = np.zeros_like(x_road)
            
            # Car is at x=0. The bump extends from t=1 to t=2. 
            # Assuming constant horizontal velocity v_x = 1 m/s for visual mapping.
            # Bump spatial start = 1.0 m, end = 2.0 m
            # Car spatial position x_car = t * v_x
            v_x = 1.0
            x_car = t * v_x
            
            for i, x in enumerate(x_road):
                # World coordinate of this point on the screen
                x_world = x_car + x
                if 1.0 <= x_world <= 2.0:
                    y_road[i] = 0.1 * np.sin(np.pi * (x_world - 1.0) / 1.0)
            
            elems['bump'].set_data(x_road, y_road)

        # Update both sides
        # Use t_eval length to prevent out-of-bounds for PINN which might be 1 frame shorter
        idx_pinn = min(frame, len(zs_pinn)-1) 
        
        update_side(elements_ode, zs_true[frame], zu_true[frame])
        update_side(elements_pinn, zs_pinn[idx_pinn], zu_pinn[idx_pinn])
        
        return [time_text, elements_ode['wheel'], elements_ode['car'], elements_ode['spring'], 
                elements_ode['damper_outer'], elements_ode['damper_inner'], elements_ode['bump'],
                elements_pinn['wheel'], elements_pinn['car'], elements_pinn['spring'], 
                elements_pinn['damper_outer'], elements_pinn['damper_inner'], elements_pinn['bump']]

    # 4. Create the animation object
    print("Rendering animation (this may take a minute to save)...")
    # Downsample frames for playback speed if necessary (e.g. 500 frames / 2 = 250 frames)
    step = 2
    frames = range(0, len(t_eval), step)
    
    ani = animation.FuncAnimation(fig, update, frames=frames, interval=20, blit=True)
    
    # Save as MP4 or GIF
    save_path = 'results/quarter_car_animation.mp4'
    try:
        writer = animation.FFMpegWriter(fps=50) # Assuming 100Hz / 2 = 50fps
        ani.save(save_path, writer=writer)
        print(f"Animation saved to {save_path}")
    except FileNotFoundError:
        print("FFMpeg not found! Falling back to GIF format...")
        save_path = 'results/quarter_car_animation.gif'
        writer = animation.PillowWriter(fps=50)
        ani.save(save_path, writer=writer)
        print(f"Animation saved to {save_path}")

    # Don't natively plt.show() because it blocks, but it's generated!

if __name__ == "__main__":
    create_animation()
