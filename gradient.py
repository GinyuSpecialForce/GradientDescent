import math
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


class LossSurface:
    # A loss surface for 2D functions L(x1, x2)
    
    def __init__(self, func_str, x1_range=(-3, 3), x2_range=(-3, 3), num_points=200):
        """
        Initialize the loss surface from a function string
        Parameters:
        - func_str: String representation of the function (e.g., "x1**2 + x2**2")
        - x1_range: Tuple of (min, max) for x1 axis
        - x2_range: Tuple of (min, max) for x2 axis
        - num_points: Number of points per axis for the grid
        """
        self.func_str = func_str
        self.x1_min, self.x1_max = x1_range
        self.x2_min, self.x2_max = x2_range
        
        # Create the grid
        x1_list = np.linspace(x1_range[0], x1_range[1], num_points)
        x2_list = np.linspace(x2_range[0], x2_range[1], num_points)
        self.X1, self.X2 = np.meshgrid(x1_list, x2_list)
        
        # Create namespace with math functions
        self.namespace = {
            'sin': math.sin,
            'cos': math.cos,
            'tan': math.tan,
            'exp': math.exp,
            'log': math.log,
            'log10': math.log10,
            'sqrt': math.sqrt,
            'pi': math.pi,
            'e': math.e,
            'abs': abs,
            'x1': 0,
            'x2': 0
        }
        
        # Evaluate the function on the grid
        self.Z = np.zeros_like(self.X1)
        for i in range(num_points):
            for j in range(num_points):
                self.namespace['x1'] = self.X1[i, j]
                self.namespace['x2'] = self.X2[i, j]
                try:
                    self.Z[i, j] = eval(func_str, {"__builtins__": {}}, self.namespace)
                except:
                    self.Z[i, j] = np.nan
    
    def plot_3d(self, trajectory=None, title=None):
        """
        Create a 3D surface plot of the loss surface with trajectory
        Parameters:
        - trajectory: List of (x1, x2) points from gradient descent
        - title: Optional title for the plot
        """
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot the surface
        surf = ax.plot_surface(self.X1, self.X2, self.Z, cmap='viridis', 
                               alpha=0.8, linewidth=0, antialiased=True)
        fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5, label='f(x)')
        
        # Overlay the trajectory if provided
        if trajectory:
            traj_x1 = [p[0] for p in trajectory]
            traj_x2 = [p[1] for p in trajectory]
            traj_z = [self.evaluate_point(p[0], p[1]) for p in trajectory]
            
            # Plot the path as a red line
            ax.plot(traj_x1, traj_x2, traj_z, 'r-', linewidth=3, label='Path')
            
            # Plot start point (green)
            ax.scatter(traj_x1[0], traj_x2[0], traj_z[0], 
                      color='green', s=100, label='Start')
            
            # Plot end point (blue)
            ax.scatter(traj_x1[-1], traj_x2[-1], traj_z[-1], 
                      color='blue', s=100, label='End')
        
        ax.set_xlabel('x1', fontsize=12)
        ax.set_ylabel('x2', fontsize=12)
        ax.set_zlabel('f(x1, x2)', fontsize=12)
        
        if title:
            ax.set_title(title, fontsize=14)
        else:
            ax.set_title(f'f(x1, x2) = {self.func_str}', fontsize=14)
        
        if trajectory:
            ax.legend()
        
        # Set viewing angle
        ax.view_init(elev=25, azim=-60)
        
        plt.tight_layout()
        return fig, ax
    
    def evaluate_point(self, x1, x2):
        # Evaluate the function at a single point
        self.namespace['x1'] = x1
        self.namespace['x2'] = x2
        return eval(self.func_str, {"__builtins__": {}}, self.namespace)


def suggest_learning_rate(func_str, start_values):
    """
    Suggest a reasonable initial learning rate based on the function structure
    Parameters:
    - func_str: The function string
    - start_values: Starting values for x1, x2, ...
    Returns:
    - suggested_lr: A safe initial learning rate
    - reason: Explanation for the suggestion
    """
    func_lower = func_str.lower()
    reason = ""
    
    # Check for high-degree polynomials
    if '**4' in func_str or '**5' in func_str or '**6' in func_str:
        suggested_lr = 0.001
        reason = "Function has degree 4 or higher (very steep gradients). Use small initial learning rate."
    elif '**3' in func_str:
        # Check if it's a cubic with large coefficients
        if '100*' in func_str or '50*' in func_str:
            suggested_lr = 0.005
            reason = "Function has large coefficients in cubic term. Use small initial learning rate."
        else:
            suggested_lr = 0.01
            reason = "Function has cubic term. Use moderate initial learning rate."
    elif '**2' in func_str or '^2' in func_str:
        # Check for large coefficients
        if '100*' in func_str or '50*' in func_str:
            suggested_lr = 0.01
            reason = "Function has large coefficients. Use moderate initial learning rate."
        else:
            suggested_lr = 0.1
            reason = "Function has quadratic term. Standard initial learning rate works."
    elif 'exp(' in func_lower or 'e**' in func_lower:
        suggested_lr = 0.001
        reason = "Function has exponential terms (very steep). Use small initial learning rate."
    elif 'sin' in func_lower or 'cos' in func_lower or 'tan' in func_lower:
        suggested_lr = 0.05
        reason = "Function has trigonometric terms. Use moderate initial learning rate."
    elif '/' in func_str:
        suggested_lr = 0.01
        reason = "Function has division (potential singularities). Use moderate initial learning rate."
    else:
        suggested_lr = 0.05
        reason = "Default safe initial learning rate."
    
    # Check starting values - if they're far from origin, reduce learning rate
    max_start = max(abs(v) for v in start_values)
    if max_start > 10:
        suggested_lr = suggested_lr / (max_start / 5)
        reason += f" Starting values are far from origin ({max_start:.1f}). Reducing initial learning rate."
    
    return suggested_lr, reason


def get_learning_rate(iteration, initial_lr, decay_type='inverse', decay_rate=1.0, power=0.5):
    """
    Calculate the learning rate at a given iteration using a diminishing schedule
    Non-summable, square-summable sequences:
    - inverse: ηₙ = η₀ / (1 + η₀ * n)  [∑η = ∞, ∑η² < ∞]
    - inverse_sqrt: ηₙ = η₀ / √(n+1)   [∑η = ∞, ∑η² < ∞]
    - inverse_power: ηₙ = η₀ / (n+1)^p where 0.5 < p ≤ 1  [∑η = ∞, ∑η² < ∞]
    
    Parameters:
    - iteration: Current iteration number (0-indexed)
    - initial_lr: Initial learning rate η₀
    - decay_type: 'inverse', 'inverse_sqrt', or 'inverse_power'
    - decay_rate: Scaling factor for decay
    - power: Power for inverse_power decay (must be > 0.5)
    Returns:
    - learning_rate: The learning rate for this iteration
    """
    n = iteration + 1  # Use 1-indexed for the formula
    
    if decay_type == 'inverse':
        # ηₙ = η₀ / (1 + η₀ * n)
        return initial_lr / (1 + initial_lr * n * decay_rate)
    
    elif decay_type == 'inverse_sqrt':
        # ηₙ = η₀ / √(n+1)
        return initial_lr / math.sqrt(n)
    
    elif decay_type == 'inverse_power':
        # ηₙ = η₀ / n^p where 0.5 < p ≤ 1
        p = max(power, 0.51)  # Ensure p > 0.5
        return initial_lr / (n ** p)
    
    else:
        # Default to inverse
        return initial_lr / (1 + initial_lr * n * decay_rate)


def plot_convergence_multi(history, var_names, title=None):
    """
    Plot the convergence of each variable over iterations (for any number of dimensions)
    Parameters:
    - history: List of variable value lists from gradient descent
    - var_names: List of variable names
    - title: Optional title for the plot
    """
    iterations = list(range(len(history)))
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Color palette for multiple variables
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD', 
              '#F39C12', '#E74C3C', '#3498DB', '#2ECC71', '#9B59B6', '#1ABC9C']
    
    for i, name in enumerate(var_names):
        values = [point[i] for point in history]
        color = colors[i % len(colors)]
        ax.plot(iterations, values, 'o-', color=color, linewidth=2, 
                markersize=3, label=name)
    
    ax.set_xlabel('Iteration', fontsize=12)
    ax.set_ylabel('Variable Value', fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best')
    
    if title:
        ax.set_title(title, fontsize=14)
    else:
        ax.set_title('Gradient Descent Convergence', fontsize=14)
    
    plt.tight_layout()
    return fig, ax


def gradient_descent_interactive():
    print("\nHow to enter multi-variable functions:")
    print("  - Use x1, x2, x3, ... for any number of variables")
    print("  - Example (2D): x1**2 + x2**2")
    print("  - Example (3D): x1**2 + x2**2 + x3**2")
    print("  - Example (4D): x1**2 + x2**2 + x3**2 + x4**2")
    print("  - Example: (x1 - 2)**2 + (x2 - 1)**2 + (x3 + 3)**2")
    print("-" * 70)
    
    # Get user input
    func_str = input("\nEnter your function in terms of x1, x2, ... : ")
    start_str = input("Enter starting values (comma-separated, e.g., 3, 4, 5): ")
    
    # Parse starting values
    start_values = [float(x.strip()) for x in start_str.split(',')]
    num_vars = len(start_values)
    
    # Create variable names dynamically
    var_names = [f'x{i+1}' for i in range(num_vars)]
    
    print(f"\n✓ Detected {num_vars} variables: {', '.join(var_names)}")
    
    # Suggest initial learning rate
    suggested_lr, reason = suggest_learning_rate(func_str, start_values)
    print(f"\n  Initial Learning Rate Suggestion: {suggested_lr}")
    print(f"   Reason: {reason}")
    
    # Get learning rate with option to use suggested
    use_suggested = input(f"\nUse suggested initial learning rate {suggested_lr}? (y/n, or enter your own): ").lower()
    if use_suggested == 'y' or use_suggested == '':
        initial_lr = suggested_lr
        print(f"  Using suggested initial learning rate: {initial_lr}")
    else:
        initial_lr = float(input("Enter initial learning rate: "))
    
    # Get decay parameters
    print("\nLearning rate decay schedule:")
    print("  1. inverse:      ηₙ = η₀ / (1 + η₀*n)  [RECOMMENDED]")
    print("  2. inverse_sqrt: ηₙ = η₀ / √n")
    print("  3. inverse_power: ηₙ = η₀ / n^p (p > 0.5)")
    
    decay_choice = input("\nChoose decay schedule (1-3, default = 1): ").strip()
    if decay_choice == '2':
        decay_type = 'inverse_sqrt'
        power = 0.5
    elif decay_choice == '3':
        decay_type = 'inverse_power'
        power = float(input("Enter power p (must be > 0.5, e.g., 0.75): "))
    else:
        decay_type = 'inverse'
        power = 1.0
    
    max_iterations = int(input("Enter maximum number of iterations: "))
    
    # Create namespace with math functions and variables
    namespace = {
        'sin': math.sin,
        'cos': math.cos,
        'tan': math.tan,
        'exp': math.exp,
        'log': math.log,
        'log10': math.log10,
        'sqrt': math.sqrt,
        'pi': math.pi,
        'e': math.e,
        'abs': abs
    }
    
    # Add variables to namespace with initial values
    for i, val in enumerate(start_values):
        namespace[f'x{i+1}'] = val
    
    # Define the function using the string
    def func(vals):
        # Update namespace with current variable values
        for i, val in enumerate(vals):
            namespace[f'x{i+1}'] = val
        return eval(func_str, {"__builtins__": {}}, namespace)
    
    # Compute gradient numerically (partial derivatives)
    def gradient(vals, h=1e-7):
        grad = []
        f_current = func(vals)
        
        for i in range(len(vals)):
            # Create a copy and perturb the i-th variable
            vals_plus = vals.copy()
            vals_plus[i] += h
            
            # Compute partial derivative using forward difference
            f_plus = func(vals_plus)
            partial = (f_plus - f_current) / h
            grad.append(partial)
        
        return grad
    
    # Run gradient descent with diminishing learning rate
    x = start_values.copy()
    history = [x.copy()]
    lr_history = [initial_lr]
    
    print(f"\nf({', '.join(var_names)}) = {func_str}")
    print(f"Start: {', '.join([f'{v:.6f}' for v in start_values])}")
    print(f"Initial Learning Rate: {initial_lr:.6f}")
    print(f"Decay Schedule: {decay_type}")
    if decay_type == 'inverse_power':
        print(f"Power: {power:.4f}")
    print(f"Max Iterations: {max_iterations}")
    print("=" * 70)
    
    # Print header
    header = f"{'Iter':<6} | "
    for name in var_names:
        header += f"{name:<14} | "
    header += f"{'f(x)':<14} | {'LR':<10}"
    print(header)
    print("-" * (6 + 16 * num_vars + 28))
    
    for i in range(max_iterations):
        grad = gradient(x)
        f_val = func(x)
        
        # Calculate learning rate for this iteration
        if decay_type == 'inverse_power':
            current_lr = get_learning_rate(i, initial_lr, decay_type, power=power)
        else:
            current_lr = get_learning_rate(i, initial_lr, decay_type)
        
        # Update rule: x_new = x_old - learning_rate * gradient
        x_new = x.copy()
        for j in range(len(x)):
            x_new[j] = x[j] - current_lr * grad[j]
        
        # Update x
        x = x_new
        f_val = func(x)
        
        # Format f(x) with scientific notation if it's very small or very large
        if abs(f_val) < 0.001 or abs(f_val) > 1000:
            f_str = f"{f_val:<14.6e}"
        else:
            f_str = f"{f_val:<14.8f}"
        
        # Print current iteration with dynamic columns
        row = f"{i:<6} | "
        for val in x:
            row += f"{val:<14.8f} | "
        row += f"{f_str} | {current_lr:<10.8f}"
        print(row)
        
        # Check for convergence
        grad_magnitude = math.sqrt(sum(g**2 for g in grad))
        if grad_magnitude < 1e-8:
            print("-" * (6 + 16 * num_vars + 28))
            print(f"  Converged after {i+1} iterations! (gradient magnitude = {grad_magnitude:.2e})")
            break
        
        # Store history
        history.append(x.copy())
        lr_history.append(current_lr)
        
        # Safety check: if numbers get too large, stop
        if any(abs(val) > 1e10 for val in x):
            print("-" * (6 + 16 * num_vars + 28))
            print("   WARNING: Values are exploding! Try a smaller initial learning rate.")
            break
    
    # Final result
    print("-" * (6 + 16 * num_vars + 28))
    print(f"FINAL RESULT:")
    for j, name in enumerate(var_names):
        print(f"  {name} = {x[j]:.10f}")
    print(f"  f(x) = {func(x):.16f}")
    print(f"  Final learning rate: {current_lr:.8f}")
    print("=" * 70)
    
    # Visualization Section
    plot_choice = input("\nWould you like to see visualizations? (y/n): ").lower()
    if plot_choice == 'y':
        
        # 3D Visualization (only for 2-dimensions)
        if num_vars == 2:
            # Determine appropriate axis ranges based on the trajectory
            x1_values = [p[0] for p in history]
            x2_values = [p[1] for p in history]
            
            # If values exploded, limit the range for visualization
            x1_max_abs = max(abs(min(x1_values)), abs(max(x1_values)))
            x2_max_abs = max(abs(min(x2_values)), abs(max(x2_values)))
            
            if x1_max_abs > 10 or x2_max_abs > 10:
                print("\n   Values exploded! Limiting visualization range to [-10, 10].")
                x1_range = (-10, 10)
                x2_range = (-10, 10)
            else:
                x1_min = min(x1_values) - 0.5
                x1_max = max(x1_values) + 0.5
                x2_min = min(x2_values) - 0.5
                x2_max = max(x2_values) + 0.5
                
                # Make sure the ranges are symmetric and reasonable
                x1_range = (min(x1_min, -1), max(x1_max, 1))
                x2_range = (min(x2_min, -1), max(x2_max, 1))
            
            print("\n[1/3] Generating 3D visualization...")
            
            # Create the loss surface
            ls = LossSurface(func_str, x1_range, x2_range)
            
            # Plot 3D surface with trajectory
            fig_3d, ax_3d = ls.plot_3d(
                trajectory=history,
                title=f'3D Gradient Descent Path\n{func_str}'
            )
            plt.show()
            
            # Save 3D figure option
            save_3d_choice = input("\nWould you like to save the 3D figure? (y/n): ").lower()
            if save_3d_choice == 'y':
                fig_3d.savefig('gradient_descent_3d.png', dpi=300, bbox_inches='tight')
                print("    Saved: gradient_descent_3d.png")
        
        # Convergence plot
        print("\n[2/3] Generating convergence plot...")
        fig, ax = plot_convergence_multi(
            history=history,
            var_names=var_names,
            title=f'Convergence of Variables\n{func_str}'
        )
        plt.show()
        
        # Save convergence figure option
        save_conv_choice = input("\nWould you like to save the convergence figure? (y/n): ").lower()
        if save_conv_choice == 'y':
            fig.savefig('gradient_descent_convergence.png', dpi=300, bbox_inches='tight')
            print("    Saved: gradient_descent_convergence.png")
        
        # Learning rate decay plot
        print("\n[3/3] Generating learning rate decay plot...")
        plt.figure(figsize=(10, 6))
        plt.plot(range(len(lr_history)), lr_history, 'b-', linewidth=2)
        plt.xlabel('Iteration', fontsize=12)
        plt.ylabel('Learning Rate', fontsize=12)
        plt.title(f'Learning Rate Decay ({decay_type})', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.yscale('log')  # Log scale makes the decay more visible
        plt.show()
        
        # Save LR figure option
        save_lr_choice = input("\nWould you like to save the learning rate decay figure? (y/n): ").lower()
        if save_lr_choice == 'y':
            plt.figure(figsize=(10, 6))
            plt.plot(range(len(lr_history)), lr_history, 'b-', linewidth=2)
            plt.xlabel('Iteration', fontsize=12)
            plt.ylabel('Learning Rate', fontsize=12)
            plt.title(f'Learning Rate Decay ({decay_type})', fontsize=14)
            plt.grid(True, alpha=0.3)
            plt.yscale('log')
            plt.savefig('learning_rate_decay.png', dpi=300, bbox_inches='tight')
            print("    Saved: learning_rate_decay.png")
            plt.close()
    
    return x, history, lr_history


if __name__ == "__main__":
    # Check if required libraries are installed
    try:
        import numpy
        import matplotlib
    except ImportError:
        print("\n   Required libraries not installed.")
        print("   Please install them with:")
        print("   pip install numpy matplotlib")
        print("\nThe gradient descent will still work, but visualizations won't.")
        print("-" * 70)
    
    gradient_descent_interactive()