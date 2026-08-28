import math
import argparse
import random
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import Normalize
import matplotlib.cm as cm


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
        
        # Evaluate the function on the grid with overflow protection
        self.Z = np.zeros_like(self.X1)
        for i in range(num_points):
            for j in range(num_points):
                self.namespace['x1'] = self.X1[i, j]
                self.namespace['x2'] = self.X2[i, j]
                try:
                    val = eval(func_str, {"__builtins__": {}}, self.namespace)
                    # Check for overflow
                    if np.isinf(val) or np.isnan(val) or abs(val) > 1e100:
                        self.Z[i, j] = np.nan
                    else:
                        self.Z[i, j] = val
                except (OverflowError, ValueError, ZeroDivisionError):
                    self.Z[i, j] = np.nan
                except Exception:
                    self.Z[i, j] = np.nan
    
    def plot_3d_simple(self, trajectory=None, title=None, save_path=None):
        """
        Create a 3D surface plot of the loss surface with a simple red trajectory.
        (No color gradient for better performance)
        
        Parameters:
        - trajectory: List of (x1, x2) points from gradient descent
        - title: Optional title for the plot
        - save_path: Optional path to save the figure
        """
        fig = plt.figure(figsize=(14, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot the surface
        Z_masked = np.ma.masked_invalid(self.Z)
        surf = ax.plot_surface(self.X1, self.X2, Z_masked, cmap='viridis', 
                               alpha=0.7, linewidth=0, antialiased=True)
        fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5, label='f(x)')
        
        # Overlay the trajectory with a simple red line
        if trajectory:
            traj_x1 = [p[0] for p in trajectory]
            traj_x2 = [p[1] for p in trajectory]
            traj_z = [self.evaluate_point(p[0], p[1]) for p in trajectory]
            
            # Only plot trajectory points that are finite
            valid_indices = [i for i, z in enumerate(traj_z) if np.isfinite(z) and abs(z) < 1e100]
            if valid_indices:
                traj_x1 = [traj_x1[i] for i in valid_indices]
                traj_x2 = [traj_x2[i] for i in valid_indices]
                traj_z = [traj_z[i] for i in valid_indices]
                
                # Plot the path as a red line
                ax.plot(traj_x1, traj_x2, traj_z, 'r-', linewidth=3, alpha=0.8, label='Path')
                
                # Plot start point (green)
                ax.scatter(traj_x1[0], traj_x2[0], traj_z[0], 
                          color='green', s=120, label='Start', edgecolor='black', linewidth=1)
                
                # Plot end point (blue)
                ax.scatter(traj_x1[-1], traj_x2[-1], traj_z[-1], 
                          color='blue', s=120, label='End', edgecolor='black', linewidth=1)
        
        ax.set_xlabel('x1', fontsize=12)
        ax.set_ylabel('x2', fontsize=12)
        ax.set_zlabel('f(x1, x2)', fontsize=12)
        
        # Set z-axis limits
        Z_finite = self.Z[np.isfinite(self.Z)]
        if len(Z_finite) > 0:
            z_min, z_max = np.percentile(Z_finite, [1, 99])
            if z_min == z_max:
                z_min, z_max = -10, 10
            ax.set_zlim(z_min, z_max)
        
        if title:
            ax.set_title(title, fontsize=14)
        else:
            ax.set_title(f'Gradient Descent Path\n{self.func_str}', fontsize=14)
        
        ax.legend()
        ax.view_init(elev=25, azim=-60)
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"    Saved: {save_path}")
        
        return fig, ax
    
    def plot_2d_with_color_gradient(self, trajectory=None, title=None, save_path=None):
        """
        Create a 2D contour plot with color gradient trajectory.
        
        Parameters:
        - trajectory: List of (x1, x2) points from gradient descent
        - title: Optional title for the plot
        - save_path: Optional path to save the figure
        """
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Plot objective contours
        Z_masked = np.ma.masked_invalid(self.Z)
        cp = ax.contour(self.X1, self.X2, Z_masked, 30, cmap='viridis', alpha=0.6)
        fig.colorbar(cp, ax=ax, label='f(x)')
        
        # Overlay the trajectory with color gradient
        if trajectory:
            traj_x1 = [p[0] for p in trajectory]
            traj_x2 = [p[1] for p in trajectory]
            
            # Use color gradient based on iteration number
            colors = cm.plasma(np.linspace(0, 1, len(traj_x1)))
            
            # Plot the path with color gradient
            for i in range(len(traj_x1) - 1):
                ax.plot(traj_x1[i:i+2], traj_x2[i:i+2], color=colors[i], 
                       linewidth=2.5, alpha=0.8)
            
            # Plot start point (green)
            ax.scatter(traj_x1[0], traj_x2[0], color='green', s=120, 
                      label='Start', edgecolor='black', linewidth=1, zorder=5)
            
            # Plot end point (red)
            ax.scatter(traj_x1[-1], traj_x2[-1], color='red', s=120, 
                      label='End', edgecolor='black', linewidth=1, zorder=5)
            
            # Add colorbar for the trajectory
            norm = Normalize(vmin=0, vmax=len(traj_x1) - 1)
            sm = plt.cm.ScalarMappable(cmap='plasma', norm=norm)
            sm.set_array([])
            cbar = fig.colorbar(sm, ax=ax, shrink=0.3, aspect=10)
            cbar.set_label('Iteration Progress', fontsize=10)
        
        ax.set_xlabel('x1', fontsize=12)
        ax.set_ylabel('x2', fontsize=12)
        ax.grid(True, alpha=0.3)
        
        if title:
            ax.set_title(title, fontsize=14)
        else:
            ax.set_title(f'Gradient Descent Path with Color Gradient\n{self.func_str}', fontsize=14)
        
        ax.legend(loc='upper right')
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"    Saved: {save_path}")
        
        return fig, ax
    
    def evaluate_point(self, x1, x2):
        # Evaluate the function at a single point with overflow protection
        try:
            self.namespace['x1'] = x1
            self.namespace['x2'] = x2
            val = eval(self.func_str, {"__builtins__": {}}, self.namespace)
            if np.isinf(val) or np.isnan(val) or abs(val) > 1e100:
                return float('inf')
            return val
        except (OverflowError, ValueError, ZeroDivisionError):
            return float('inf')
        except Exception as e:
            print(f"   Error evaluating at ({x1}, {x2}): {e}")
            return float('inf')


def parse_arguments():
    # Parse command-line arguments for non-interactive mode
    parser = argparse.ArgumentParser(
        description='Gradient Descent Visualizer with Color Gradient',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic run with color gradient
  python gradient.py -f "x1**2 + x2**2" -s "3,4" -lr 0.1 -i 50
  
  # Multi-start with color gradient
  python gradient.py -f "sin(x1)*cos(x2)" --multi 10 --range=-5,5
        """
    )
    parser.add_argument(
        '-f', '--function',
        type=str,
        help='Function to minimize (e.g., "x1**2 + x2**2")'
    )
    parser.add_argument(
        '-s', '--start',
        type=str,
        help='Starting values, comma-separated (e.g., "3, 4")'
    )
    parser.add_argument(
        '-lr', '--learning_rate',
        type=float,
        help='Initial learning rate (if not provided, will be suggested)'
    )
    parser.add_argument(
        '-d', '--decay',
        type=str,
        choices=['inverse', 'inverse_sqrt', 'inverse_power'],
        default='inverse',
        help='Decay schedule: inverse, inverse_sqrt, or inverse_power (default: inverse)'
    )
    parser.add_argument(
        '-p', '--power',
        type=float,
        default=0.75,
        help='Power for inverse_power decay (must be > 0.5, default: 0.75)'
    )
    parser.add_argument(
        '-i', '--iterations',
        type=int,
        default=100,
        help='Maximum number of iterations per run (default: 100)'
    )
    parser.add_argument(
        '--no-plots',
        action='store_true',
        help='Disable plotting (run in headless mode)'
    )
    parser.add_argument(
        '--save',
        type=str,
        metavar='PREFIX',
        help='Save plots to files with this prefix (e.g., "my_run" saves as my_run_3d.png)'
    )
    parser.add_argument(
        '--multi',
        type=int,
        metavar='N',
        help='Multi-start: run gradient descent N times from different starting points'
    )
    parser.add_argument(
        '--range',
        type=str,
        default='-5,5',
        help='Range for random starting points in multi-start mode. Use --range=-5,5'
    )
    parser.add_argument(
        '--noise',
        type=float,
        metavar='AMOUNT',
        help='Add random noise to learning rate at intervals (e.g., 0.5)'
    )
    parser.add_argument(
        '--noise_freq',
        type=int,
        default=10,
        help='Frequency of noise injection (every N iterations, default: 10)'
    )
    return parser.parse_args()


def safe_get_input(prompt, input_type=str, validation=None, error_msg="Invalid input. Try again."):
    """
    Safely get user input with validation and retry
    Parameters:
    - prompt: The prompt to display
    - input_type: Type to convert to (str, float, int)
    - validation: Optional function that returns True for valid input
    - error_msg: Message to show on error
    Returns:
    - Validated user input
    """
    while True:
        try:
            user_input = input(prompt)
            if input_type != str:
                user_input = input_type(user_input)
            if validation and not validation(user_input):
                print(error_msg)
                continue
            return user_input
        except ValueError:
            print(f"  Invalid input. Expected {input_type.__name__}. Try again.")


def validate_function(func_str, namespace, num_vars=2):
    """
    Validate that the function string is syntactically correct
    Parameters:
    - func_str: The function string
    - namespace: The namespace with math functions
    - num_vars: Number of variables to test
    Returns:
    - True if valid, False otherwise
    """
    test_vals = [1.0] * num_vars
    try:
        for i, val in enumerate(test_vals):
            namespace[f'x{i+1}'] = val
        result = eval(func_str, {"__builtins__": {}}, namespace)
        if not isinstance(result, (int, float)):
            print(f"   Function returned {type(result)} instead of number")
            return False
        return True
    except Exception as e:
        print(f"   Function validation failed: {e}")
        return False


def suggest_learning_rate(func_str, start_values):
    # Suggest a reasonable initial learning rate based on the function structure
    func_lower = func_str.lower()
    reason = ""
    
    # Remove spaces for easier checking
    clean_func = func_str.replace(' ', '')
    
    # Check for high-degree polynomials - improved detection
    if '**4' in clean_func or '**5' in clean_func or '**6' in clean_func:
        suggested_lr = 0.00001  # Very small for degree 4+
        reason = "Function has degree 4 or higher (very steep gradients). Use very small initial learning rate."
    elif '**3' in clean_func:
        suggested_lr = 0.0001  # Smaller for cubic
        reason = "Function has cubic term. Use small initial learning rate."
    elif '**2' in clean_func or '^2' in clean_func:
        # Check for large coefficients or large constant shifts (e.g., (x-5)**2)
        if '100*' in clean_func or '50*' in clean_func:
            suggested_lr = 0.001
            reason = "Function has large coefficients. Use small initial learning rate."
        else:
            # Check if there's a large number inside parentheses (e.g., (x-5)**2)
            import re
            # Find numbers in the function
            numbers = re.findall(r'[-+]?\d*\.?\d+', clean_func)
            max_num = 0
            for num in numbers:
                try:
                    val = abs(float(num))
                    if val > max_num and val < 1000:  # Ignore very large constants
                        max_num = val
                except:
                    pass
            
            if max_num > 10:
                suggested_lr = 0.001
                reason = f"Function has large constant shift ({max_num:.0f}). Use small initial learning rate."
            else:
                suggested_lr = 0.1
                reason = "Function has quadratic term. Standard initial learning rate works."
    elif 'exp(' in func_lower or 'e**' in func_lower:
        suggested_lr = 0.00001
        reason = "Function has exponential terms (very steep). Use very small initial learning rate."
    elif 'sin' in func_lower or 'cos' in func_lower or 'tan' in func_lower:
        suggested_lr = 0.01
        reason = "Function has trigonometric terms. Use moderate initial learning rate."
    else:
        suggested_lr = 0.01
        reason = "Default safe initial learning rate."
    
    # Check starting values - if they're far from origin, reduce learning rate
    max_start = max(abs(v) for v in start_values)
    if max_start > 10:
        suggested_lr = suggested_lr / (max_start / 5)
        reason += f" Starting values are far from origin ({max_start:.1f}). Reducing initial learning rate."
    
    # Additional safety: ensure LR is not too large
    suggested_lr = min(suggested_lr, 0.1)
    
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


def run_single_gradient_descent(func_str, start_values, initial_lr, decay_type, power, max_iterations, 
                                 noise_amount=0, noise_freq=10, verbose=True, 
                                 clamp_range=None):
    """
    Run a single gradient descent instance, optionally with noise
    Parameters:
    - clamp_range: Tuple of (min, max) to clamp values within range
    """
    num_vars = len(start_values)
    var_names = [f'x{i+1}' for i in range(num_vars)]
    
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
    
    # Define the function using the string with error handling
    def func(vals):
        try:
            # Update namespace with current variable values
            for i, val in enumerate(vals):
                namespace[f'x{i+1}'] = val
            val = eval(func_str, {"__builtins__": {}}, namespace)
            if np.isinf(val) or np.isnan(val) or abs(val) > 1e100:
                return float('inf')
            return val
        except (OverflowError, ValueError, ZeroDivisionError):
            return float('inf')
        except Exception:
            return float('inf')
    
    # Compute gradient numerically (partial derivatives)
    def gradient(vals, h=1e-7):
        grad = []
        f_current = func(vals)
        
        # If f_current is infinite, return large gradient to move away
        if np.isinf(f_current):
            return [1e6] * len(vals)
        
        for i in range(len(vals)):
            # Create a copy and perturb the i-th variable
            vals_plus = vals.copy()
            vals_plus[i] += h
            
            # Compute partial derivative using forward difference
            f_plus = func(vals_plus)
            partial = (f_plus - f_current) / h
            
            # Clip gradient to prevent explosion
            if np.isinf(partial) or np.isnan(partial):
                partial = 1e6 if f_plus > f_current else -1e6
            
            # Hard limit on gradient magnitude
            if abs(partial) > 1e6:
                partial = math.copysign(1e6, partial)
            grad.append(partial)
        
        return grad
    
    # Run gradient descent
    x = start_values.copy()
    history = [x.copy()]
    lr_history = [initial_lr]
    current_lr = initial_lr
    noise_used = 0
    clamp_count = 0  # Track total clamps
    clamp_history = [0]  # Track clamps per iteration
    
    # Convergence tracking
    converged = False
    iterations_to_converge = max_iterations
    final_grad_magnitude = 0
    
    # Parse clamp range if provided
    clamp_min = None
    clamp_max = None
    clamp_active = False
    if clamp_range:
        range_parts = clamp_range.split(',')
        clamp_min = float(range_parts[0].strip())
        clamp_max = float(range_parts[1].strip())
        clamp_active = True
    
    for i in range(max_iterations):
        grad = gradient(x)
        f_val = func(x)
        
        # If function is infinite, try smaller steps
        if np.isinf(f_val):
            current_lr = current_lr * 0.5
            if verbose:
                print(f"         Infinite value! Reducing LR to {current_lr:.8f}")
            if current_lr < 1e-12:
                if verbose:
                    print("         Learning rate too small. Stopping.")
                break
            continue
        
        # Calculate base learning rate for this iteration
        if decay_type == 'inverse_power':
            base_lr = get_learning_rate(i, initial_lr, decay_type, power=power)
        else:
            base_lr = get_learning_rate(i, initial_lr, decay_type)
        
        # Use the smaller of base_lr and current_lr (for safety)
        actual_lr = min(base_lr, current_lr * 1.5) if i > 0 else base_lr
        
        # Apply noise if enabled
        if noise_amount > 0 and i > 0 and i % noise_freq == 0:
            # Add random noise to learning rate
            noise = random.uniform(-noise_amount, noise_amount)
            actual_lr = max(0, actual_lr + noise * actual_lr)
            noise_used += 1
            if verbose:
                print(f"        Noise injected at iter {i}: LR {base_lr:.6f} -> {actual_lr:.6f}")
        
        # Update rule: x_new = x_old - learning_rate * gradient
        x_new = x.copy()
        iteration_clamps = 0
        
        for j in range(len(x)):
            update = actual_lr * grad[j]
            # Limit maximum step size
            if abs(update) > 100:
                update = math.copysign(100, update)
            x_new[j] = x[j] - update
            
            # Clamp to range if provided
            if clamp_min is not None and clamp_max is not None:
                if x_new[j] < clamp_min:
                    x_new[j] = clamp_min
                    iteration_clamps += 1
                    if verbose:
                        print(f"        Clamped {var_names[j]} to {clamp_min:.4f}")
                elif x_new[j] > clamp_max:
                    x_new[j] = clamp_max
                    iteration_clamps += 1
                    if verbose:
                        print(f"        Clamped {var_names[j]} to {clamp_max:.4f}")
        
        clamp_count += iteration_clamps
        clamp_history.append(iteration_clamps)
        
        # Check if update made things worse (function increased)
        f_new = func(x_new)
        if f_new > f_val and f_val > 0.001:
            # Reduce learning rate and try again
            actual_lr = actual_lr * 0.5
            if verbose:
                print(f"         Step too large! Reducing LR to {actual_lr:.8f}")
            x_new = x.copy()
            iteration_clamps = 0
            
            for j in range(len(x)):
                update = actual_lr * grad[j]
                if abs(update) > 100:
                    update = math.copysign(100, update)
                x_new[j] = x[j] - update
                
                # Clamp to range if provided
                if clamp_min is not None and clamp_max is not None:
                    if x_new[j] < clamp_min:
                        x_new[j] = clamp_min
                        iteration_clamps += 1
                    elif x_new[j] > clamp_max:
                        x_new[j] = clamp_max
                        iteration_clamps += 1
            
            clamp_count += iteration_clamps
        
        # Update x
        x = x_new
        f_val = f_new
        
        # Print current iteration if verbose
        if verbose:
            # Format f(x) with scientific notation if it's very small or very large
            if abs(f_val) < 0.001 or abs(f_val) > 1000:
                f_str = f"{f_val:<14.6e}"
            else:
                f_str = f"{f_val:<14.8f}"
            
            # Print current iteration with dynamic columns
            row = f"{i:<6} | "
            for val in x:
                row += f"{val:<14.8f} | "
            row += f"{f_str} | {actual_lr:<10.8f}"
            if noise_amount > 0 and i > 0 and i % noise_freq == 0:
                row += "  "
            if clamp_active and iteration_clamps > 0:
                row += f"  {iteration_clamps}"
            print(row)
        
        # Check for convergence
        grad_magnitude = math.sqrt(sum(g**2 for g in grad))
        final_grad_magnitude = grad_magnitude
        if grad_magnitude < 1e-8:
            converged = True
            iterations_to_converge = i + 1
            if verbose:
                print("-" * (6 + 16 * num_vars + 28))
                print(f"    Converged after {i+1} iterations! (gradient magnitude = {grad_magnitude:.2e})")
            break
        
        # Store history
        history.append(x.copy())
        lr_history.append(actual_lr)
        
        # Safety check: if numbers get too large, stop
        if any(abs(val) > 1e10 for val in x):
            if verbose:
                print("-" * (6 + 16 * num_vars + 28))
                print("      WARNING: Values are exploding! Try a smaller initial learning rate.")
            break
    
    final_f = func(x)
    return x, history, lr_history, final_f, noise_used, clamp_count, clamp_history, clamp_active, converged, iterations_to_converge, final_grad_magnitude


def run_multi_start_gradient_descent(func_str, start_values, initial_lr, decay_type, power, 
                                      max_iterations, num_starts, range_str, 
                                      noise_amount=0, noise_freq=10, show_plots=True, save_prefix=None):
    # Run gradient descent multiple times from different starting points
    # Parse range
    range_parts = range_str.split(',')
    min_val = float(range_parts[0].strip())
    max_val = float(range_parts[1].strip())
    num_vars = len(start_values)
    var_names = [f'x{i+1}' for i in range(num_vars)]
    
    print(f"\n  Multi-start: Running {num_starts} gradient descents")
    print(f"   Random start range: [{min_val}, {max_val}]")
    print(f"   {'With noise' if noise_amount > 0 else 'No noise'}")
    print("=" * 70)
    
    # Keep track of best result
    best_f = float('inf')
    best_x = None
    best_history = None
    best_lr_history = None
    best_clamp_count = 0
    best_converged = False
    best_iterations = 0
    best_grad_mag = 0
    best_run_idx = -1
    all_results = []
    
    # Track best non-clamping run
    best_non_clamp_f = float('inf')
    best_non_clamp_x = None
    best_non_clamp_history = None
    best_non_clamp_lr_history = None
    
    total_clamps = 0
    clamp_active = True
    
    # Progress bar for multi-start
    use_progress = False
    pbar = None
    if num_starts > 10:
        try:
            from tqdm import tqdm
            pbar = tqdm(total=num_starts, desc="Multi-start progress")
            use_progress = True
        except ImportError:
            print(f"  Running {num_starts} starts (no progress bar - install tqdm for progress)")
    
    for run_num in range(num_starts):
        # Generate random starting point
        if run_num == 0:
            current_start = start_values.copy()
        else:
            current_start = [random.uniform(min_val, max_val) for _ in range(num_vars)]
        
        if not use_progress:
            print(f"\n--- Run {run_num + 1}/{num_starts} ---")
            print(f"Start: {', '.join([f'{v:.4f}' for v in current_start])}")
        
        # Run gradient descent with clamp range
        x, history, lr_history, final_f, noise_used, clamp_count, clamp_history, clamp_active_local, converged, iters, grad_mag = run_single_gradient_descent(
            func_str, current_start, initial_lr, decay_type, power, max_iterations,
            noise_amount, noise_freq, verbose=False,
            clamp_range=range_str
        )
        
        total_clamps += clamp_count
        
        # Print summary for this run
        if not use_progress:
            clamp_info = f"    Clamped {clamp_count} times" if clamp_count > 0 else "    No clamping needed"
            converge_info = f"Converged: {'Yes' if converged else 'No'}"
            print(f"  Final: {', '.join([f'{v:.4f}' for v in x])}  |  f(x) = {final_f:.6f}")
            print(f"  {clamp_info}" + (f"    Noise used {noise_used} times" if noise_used > 0 else ""))
            print(f"  {converge_info} ({iters} iterations, grad = {grad_mag:.2e})")
        
        # Store result
        result = {
            'start': current_start,
            'end': x,
            'final_f': final_f,
            'history': history,
            'lr_history': lr_history,
            'noise_used': noise_used,
            'clamp_count': clamp_count,
            'clamp_history': clamp_history,
            'clamp_active': clamp_active_local,
            'run_num': run_num + 1,
            'converged': converged,
            'iterations': iters,
            'grad_magnitude': grad_mag
        }
        all_results.append(result)
        
        # Update best overall
        if final_f < best_f:
            best_f = final_f
            best_x = x
            best_history = history
            best_lr_history = lr_history
            best_clamp_count = clamp_count
            best_converged = converged
            best_iterations = iters
            best_grad_mag = grad_mag
            best_run_idx = run_num
            if not use_progress:
                print(f"      NEW BEST: f(x) = {best_f:.6f}")
        
        # Update best non-clamping run
        if clamp_count == 0 and final_f < best_non_clamp_f:
            best_non_clamp_f = final_f
            best_non_clamp_x = x
            best_non_clamp_history = history
            best_non_clamp_lr_history = lr_history
            if final_f < best_f and not use_progress:
                print(f"      BEST NON-CLAMPING: f(x) = {best_non_clamp_f:.6f}")
        
        # Update progress bar
        if use_progress and pbar is not None:
            pbar.update(1)
            pbar.set_postfix({'best': f'{best_f:.6f}'})
    
    if use_progress and pbar is not None:
        pbar.close()
    
    # Print summary
    print("\nMulti-start Summary")
    print("-" * 70)
    print(f"Best f(x): {best_f:.10f}")
    print(f"Best x: {', '.join([f'{v:.6f}' for v in best_x])}")
    if best_run_idx >= 0:
        print(f"Best run: {best_run_idx + 1}/{num_starts}")
    print(f"Converged: {'Yes' if best_converged else 'No'}")
    print(f"Iterations to converge: {best_iterations}")
    print(f"Final gradient magnitude: {best_grad_mag:.2e}")
    print(f"Runs completed: {num_starts}")
    print(f"Total clamping events: {total_clamps}")
    if total_clamps > 0:
        print(f"     Some values were clamped to stay within [{min_val}, {max_val}]")
        print(f"    Try increasing the range or using a smaller learning rate")
        if best_non_clamp_f < float('inf'):
            print(f"    Best non-clamping run: f(x) = {best_non_clamp_f:.10f}")
            print(f"     x: {', '.join([f'{v:.6f}' for v in best_non_clamp_x])}")
    
    # --- Summary plots - ONLY best overall and best non-clamping ---
    if show_plots:
        print("\n" + "-" * 70)
        print("Summary Plots")
        print("-" * 70)
        
        # 1. Best run convergence plot
        print("\n[1/5] Generating best run convergence plot...")
        fig_best, ax_best = plt.subplots(figsize=(12, 6))
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD', 
                  '#F39C12', '#E74C3C', '#3498DB', '#2ECC71', '#9B59B6', '#1ABC9C']
        
        iterations = list(range(len(best_history)))
        
        for i, name in enumerate(var_names):
            values = [point[i] for point in best_history]
            color = colors[i % len(colors)]
            ax_best.plot(iterations, values, 'o-', color=color, linewidth=2, 
                        markersize=3, label=name)
        
        # Add clamping markers to the convergence plot if clamping occurred
        if best_clamp_count > 0:
            for result in all_results:
                if result['history'] == best_history:
                    clamp_hist = result.get('clamp_history', [])
                    if clamp_hist:
                        clamp_iters = [i for i, count in enumerate(clamp_hist) if count > 0]
                        if clamp_iters:
                            y_vals = [best_history[i][0] for i in clamp_iters if i < len(best_history)]
                            ax_best.scatter(clamp_iters, y_vals, color='red', s=80, 
                                          marker='v', zorder=5, label='Clamping occurred')
        
        ax_best.set_xlabel('Iteration', fontsize=12)
        ax_best.set_ylabel('Variable Value', fontsize=12)
        ax_best.grid(True, alpha=0.3)
        ax_best.legend(loc='best')
        ax_best.axhline(y=min_val, color='red', linestyle='--', alpha=0.5, label=f'Range boundary: {min_val}')
        ax_best.axhline(y=max_val, color='red', linestyle='--', alpha=0.5, label=f'Range boundary: {max_val}')
        
        title_text = f'Best Run Convergence (f(x) = {best_f:.6f})\n{func_str}'
        if best_clamp_count > 0:
            title_text += f'\n  Clamping events: {best_clamp_count}'
        if best_converged:
            title_text += f'\n  Converged after {best_iterations} iterations'
        ax_best.set_title(title_text, fontsize=14)
        plt.tight_layout()
        
        if save_prefix:
            fig_best.savefig(f'{save_prefix}_best_convergence.png', dpi=300, bbox_inches='tight')
            print(f"    Saved: {save_prefix}_best_convergence.png")
        else:
            plt.show()
        
        # 2. Best run learning rate decay plot
        print("\n[2/5] Generating best run learning rate decay plot...")
        fig_lr, ax_lr = plt.subplots(figsize=(10, 6))
        ax_lr.plot(range(len(best_lr_history)), best_lr_history, 'b-', linewidth=2)
        ax_lr.set_xlabel('Iteration', fontsize=12)
        ax_lr.set_ylabel('Learning Rate', fontsize=12)
        ax_lr.set_title(f'Learning Rate Decay - Best Run (f(x) = {best_f:.6f})\n{decay_type}', fontsize=14)
        ax_lr.grid(True, alpha=0.3)
        ax_lr.set_yscale('log')
        plt.tight_layout()
        
        if save_prefix:
            fig_lr.savefig(f'{save_prefix}_best_lr_decay.png', dpi=300, bbox_inches='tight')
            print(f"    Saved: {save_prefix}_best_lr_decay.png")
            plt.close(fig_lr)
        else:
            plt.show()
        
        # 3. Best non-clamping run convergence plot (only if exists)
        if best_non_clamp_history is not None:
            print("\n[3/5] Generating best non-clamping run convergence plot...")
            fig_best_nc, ax_best_nc = plt.subplots(figsize=(12, 6))
            
            for i, name in enumerate(var_names):
                values = [point[i] for point in best_non_clamp_history]
                color = colors[i % len(colors)]
                ax_best_nc.plot(range(len(best_non_clamp_history)), values, 'o-', color=color, linewidth=2, 
                               markersize=3, label=name)
            
            ax_best_nc.set_xlabel('Iteration', fontsize=12)
            ax_best_nc.set_ylabel('Variable Value', fontsize=12)
            ax_best_nc.grid(True, alpha=0.3)
            ax_best_nc.legend(loc='best')
            ax_best_nc.axhline(y=min_val, color='red', linestyle='--', alpha=0.5, label=f'Range boundary: {min_val}')
            ax_best_nc.axhline(y=max_val, color='red', linestyle='--', alpha=0.5, label=f'Range boundary: {max_val}')
            
            title_text = f'Best Non-Clamping Run Convergence (f(x) = {best_non_clamp_f:.6f})\n{func_str}'
            ax_best_nc.set_title(title_text, fontsize=14)
            plt.tight_layout()
            
            if save_prefix:
                fig_best_nc.savefig(f'{save_prefix}_best_non_clamp_convergence.png', dpi=300, bbox_inches='tight')
                print(f"    Saved: {save_prefix}_best_non_clamp_convergence.png")
            else:
                plt.show()
            
            # 4. Best non-clamping run learning rate decay plot
            print("\n[4/5] Generating best non-clamping run learning rate decay plot...")
            fig_lr_nc, ax_lr_nc = plt.subplots(figsize=(10, 6))
            ax_lr_nc.plot(range(len(best_non_clamp_lr_history)), best_non_clamp_lr_history, 'b-', linewidth=2)
            ax_lr_nc.set_xlabel('Iteration', fontsize=12)
            ax_lr_nc.set_ylabel('Learning Rate', fontsize=12)
            ax_lr_nc.set_title(f'Learning Rate Decay - Best Non-Clamping Run (f(x) = {best_non_clamp_f:.6f})\n{decay_type}', fontsize=14)
            ax_lr_nc.grid(True, alpha=0.3)
            ax_lr_nc.set_yscale('log')
            plt.tight_layout()
            
            if save_prefix:
                fig_lr_nc.savefig(f'{save_prefix}_best_non_clamp_lr_decay.png', dpi=300, bbox_inches='tight')
                print(f"    Saved: {save_prefix}_best_non_clamp_lr_decay.png")
                plt.close(fig_lr_nc)
            else:
                plt.show()
        else:
            print("\n[3/5] No non-clamping run found - skipping...")
            print("[4/5] No non-clamping run found - skipping...")
        
        # 5. 3D Visualization with all trajectories (only for 2D)
        if num_vars == 2:
            all_x1 = []
            all_x2 = []
            for result in all_results:
                for point in result['history']:
                    all_x1.append(point[0])
                    all_x2.append(point[1])
            
            if all_x1 and all_x2:
                x1_range = (min_val - 0.5, max_val + 0.5)
                x2_range = (min_val - 0.5, max_val + 0.5)
                
                print("\n[5/5] Generating 3D visualization...")
                
                ls = LossSurface(func_str, x1_range, x2_range)
                
                # Use the best trajectory for the simple 3D plot
                if best_history:
                    fig_3d, ax_3d = ls.plot_3d_simple(
                        trajectory=best_history,
                        title=f'Best Run: f(x) = {best_f:.6f}\n{func_str}',
                        save_path=f'{save_prefix}_3d.png' if save_prefix else None
                    )
                    plt.show()
                
                # Also show 2D color gradient
                if best_history:
                    fig_2d, ax_2d = ls.plot_2d_with_color_gradient(
                        trajectory=best_history,
                        title=f'Best Run: f(x) = {best_f:.6f}\n{func_str}',
                        save_path=f'{save_prefix}_2d_color.png' if save_prefix else None
                    )
                    plt.show()
        else:
            print("\n[5/5] Skipping 3D visualization (requires 2 variables)")
    
    return best_x, best_f, all_results


def run_gradient_descent(func_str, start_values, initial_lr, decay_type, power, max_iterations, 
                         show_plots=True, save_prefix=None,
                         multi_start=False, num_starts=1, range_str="-5,5",
                         noise_amount=0, noise_freq=10):
    """
    Run gradient descent with optional multi-start and noise
    Parameters:
    - All parameters from run_single_gradient_descent, plus:
    - multi_start: Whether to use multi-start
    - num_starts: Number of starts for multi-start
    - range_str: Range for random starts
    - noise_amount: Amount of random noise to add to LR
    - noise_freq: How often to inject noise
    """
    
    num_vars = len(start_values)
    var_names = [f'x{i+1}' for i in range(num_vars)]
    
    # If multi-start is enabled
    if multi_start and num_starts > 1:
        return run_multi_start_gradient_descent(
            func_str, start_values, initial_lr, decay_type, power, max_iterations,
            num_starts, range_str, noise_amount, noise_freq, show_plots, save_prefix
        )
    
    # Single run with optional noise and clamping
    print(f"\nf({', '.join(var_names)}) = {func_str}")
    print(f"Start: {', '.join([f'{v:.6f}' for v in start_values])}")
    print(f"Initial Learning Rate: {initial_lr:.6f}")
    print(f"Decay Schedule: {decay_type}")
    if decay_type == 'inverse_power':
        print(f"Power: {power:.4f}")
    print(f"Max Iterations: {max_iterations}")
    if noise_amount > 0:
        print(f"Noise: {noise_amount} (every {noise_freq} iterations)")
    
    # Check if we should use clamping for single run
    clamp_range = None
    if range_str != "-5,5":
        clamp_range = range_str
        print(f"Range: {range_str} (values will be clamped to this range)")
    print("-" * 70)
    
    # Print header
    header = f"{'Iter':<6} | "
    for name in var_names:
        header += f"{name:<14} | "
    header += f"{'f(x)':<14} | {'LR':<10}"
    print(header)
    print("-" * (6 + 16 * num_vars + 28))
    
    # Run single gradient descent
    x, history, lr_history, final_f, noise_used, clamp_count, clamp_history, clamp_active, converged, iterations_to_converge, final_grad_magnitude = run_single_gradient_descent(
        func_str, start_values, initial_lr, decay_type, power, max_iterations,
        noise_amount, noise_freq, verbose=True,
        clamp_range=clamp_range
    )
    
    # Final result
    print("-" * (6 + 16 * num_vars + 28))
    print(f"FINAL RESULT:")
    for j, name in enumerate(var_names):
        print(f"  {name} = {x[j]:.10f}")
    print(f"  f(x) = {final_f:.16f}")
    print(f"  Final learning rate: {lr_history[-1]:.8f}")
    if noise_used > 0:
        print(f"  Noise injections: {noise_used}")
    if clamp_count > 0:
        print(f"    Clamping events: {clamp_count}")
    # Convergence statistics
    if converged:
        print(f"  ✓ Converged after {iterations_to_converge} iterations")
    else:
        print(f"  ✗ Did not converge (reached max iterations)")
    print(f"  Final gradient magnitude: {final_grad_magnitude:.2e}")
    print("-" * 70)
    
    # Show individual graphs for single run
    if show_plots:
        # Determine appropriate axis ranges for 2D plots
        if num_vars == 2:
            x1_values = [p[0] for p in history]
            x2_values = [p[1] for p in history]
            
            x1_min = min(x1_values) - 0.5
            x1_max = max(x1_values) + 0.5
            x2_min = min(x2_values) - 0.5
            x2_max = max(x2_values) + 0.5
            
            x1_range = (min(x1_min, -1), max(x1_max, 1))
            x2_range = (min(x2_min, -1), max(x2_max, 1))
            
            # Create loss surface
            ls = LossSurface(func_str, x1_range, x2_range)
            
            # 1. 3D Simple Plot (no color gradient)
            print("\n[1/4] Generating 3D visualization...")
            fig_3d, ax_3d = ls.plot_3d_simple(
                trajectory=history,
                title=f'Gradient Descent Path\n{func_str}',
                save_path=f'{save_prefix}_3d.png' if save_prefix else None
            )
            plt.show()
            
            # 2. 2D Color Gradient Plot
            print("\n[2/4] Generating 2D color gradient visualization...")
            fig_2d, ax_2d = ls.plot_2d_with_color_gradient(
                trajectory=history,
                title=f'Gradient Descent Path with Color Gradient\n{func_str}',
                save_path=f'{save_prefix}_2d_color.png' if save_prefix else None
            )
            plt.show()
            
            # 3. Convergence plot
            print("\n[3/4] Generating convergence plot...")
            fig_conv, ax_conv = plot_convergence_multi(
                history=history,
                var_names=var_names,
                title=f'Convergence of Variables\n{func_str}'
            )
            
            if save_prefix:
                fig_conv.savefig(f'{save_prefix}_convergence.png', dpi=300, bbox_inches='tight')
                print(f"    Saved: {save_prefix}_convergence.png")
            else:
                plt.show()
            
            # 4. Learning rate decay plot
            print("\n[4/4] Generating learning rate decay plot...")
            fig_lr, ax_lr = plt.subplots(figsize=(10, 6))
            ax_lr.plot(range(len(lr_history)), lr_history, 'b-', linewidth=2)
            ax_lr.set_xlabel('Iteration', fontsize=12)
            ax_lr.set_ylabel('Learning Rate', fontsize=12)
            ax_lr.set_title(f'Learning Rate Decay ({decay_type})', fontsize=14)
            ax_lr.grid(True, alpha=0.3)
            ax_lr.set_yscale('log')
            plt.tight_layout()
            
            if save_prefix:
                fig_lr.savefig(f'{save_prefix}_lr_decay.png', dpi=300, bbox_inches='tight')
                print(f"    Saved: {save_prefix}_lr_decay.png")
                plt.close(fig_lr)
            else:
                plt.show()
    
    return x, history, lr_history


def gradient_descent_interactive(args=None):
    """
    Run gradient descent, either interactively or from command-line arguments
    Parameters:
    - args: Parsed command-line arguments (optional)
    """
    if args and args.function and args.start:
        # Command-line mode
        func_str = args.function
        start_values = [float(x.strip()) for x in args.start.split(',')]
        decay_type = args.decay
        power = args.power
        max_iterations = args.iterations
        show_plots = not args.no_plots
        save_prefix = args.save
        
        # Multi-start settings
        multi_start = args.multi is not None and args.multi > 1
        num_starts = args.multi if multi_start else 1
        range_str = args.range
        
        # Noise settings
        noise_amount = args.noise if args.noise is not None else 0
        noise_freq = args.noise_freq
        
        # Suggest LR if not provided
        if args.learning_rate is None:
            initial_lr, reason = suggest_learning_rate(func_str, start_values)
            print(f"  Suggested learning rate: {initial_lr} (Reason: {reason})")
        else:
            initial_lr = args.learning_rate
        
        print(f"\n  Running gradient descent in command-line mode...")
        print(f"   Function: {func_str}")
        print(f"   Start: {start_values}")
        print(f"   Learning Rate: {initial_lr}")
        print(f"   Decay: {decay_type}")
        print(f"   Iterations: {max_iterations}")
        if multi_start:
            print(f"   Multi-start: {num_starts} runs, range: {range_str}")
        if noise_amount > 0:
            print(f"   Noise: {noise_amount} (every {noise_freq} iterations)")
        print("-" * 70)
        
        # Run gradient descent
        return run_gradient_descent(
            func_str, start_values, initial_lr, decay_type, 
            power, max_iterations, show_plots, save_prefix,
            multi_start, num_starts, range_str,
            noise_amount, noise_freq
        )
    
    else:
        # Interactive mode
        print("\nHow to enter multi-variable functions:")
        print("  - Use x1, x2, x3, ... for any number of variables")
        print("  - Example (2D): x1**2 + x2**2")
        print("  - Example (3D): x1**2 + x2**2 + x3**2")
        print("  - Example: (x1 - 2)**2 + (x2 - 1)**2")
        print("-" * 70)
        
        # Get user input with validation
        func_str = safe_get_input(
            "\nEnter your function in terms of x1, x2, ... : ",
            validation=lambda s: any(var in s for var in ['x1', 'x2']),
            error_msg="Function must contain x1 or x2. Try again."
        )
        
        start_str = safe_get_input(
            "Enter starting values (comma-separated, e.g., 3, 4): ",
            validation=lambda s: len(s.split(',')) >= 2 and all(
                x.strip().replace('-','').replace('.','').isdigit() 
                for x in s.split(',')
            ),
            error_msg="Invalid numbers. Use comma-separated numbers (e.g., 3, 4)"
        )
        
        # Parse starting values
        start_values = [float(x.strip()) for x in start_str.split(',')]
        num_vars = len(start_values)
        var_names = [f'x{i+1}' for i in range(num_vars)]
        
        print(f"\n    Detected {num_vars} variables: {', '.join(var_names)}")
        
        print("\n" + "-" * 70)
        print("Local Minima Escape Strategies")
        print("-" * 70)
        print("  You can enable these to help find the global minimum:")
        print("  1. Multi-start: Run GD from multiple random starting points")
        print("  2. Noise: Add random jumps to the learning rate to escape local minima")
        print("  3. Neither: Just run a single GD from your starting point")
        print("  (Note: You can combine Multi-start with Noise by choosing option 1)")
        
        strategy_choice = safe_get_input(
            "\nChoose strategy (1 = Multi-start, 2 = Noise, 3 = Neither, default = 3): ",
            validation=lambda x: x in ['1', '2', '3', ''],
            error_msg="Enter 1, 2, 3, or press Enter for default."
        )
        
        multi_start = False
        num_starts = 1
        range_str = "-5,5"
        noise_amount = 0
        noise_freq = 10
        
        if strategy_choice == '1':
            # Multi-start (with optional noise)
            multi_start = True
            num_starts = safe_get_input(
                "Number of random starts (e.g., 10): ",
                input_type=int,
                validation=lambda x: x > 1,
                error_msg="Must be greater than 1."
            )
            range_str = safe_get_input(
                "Range for random starts (e.g., -5,5 or -10,10): ",
                validation=lambda s: len(s.split(',')) == 2 and all(
                    x.strip().replace('-','').replace('.','').isdigit() 
                    for x in s.split(',')
                ),
                error_msg="Enter two numbers separated by comma (e.g., -5,5)"
            )
            
            # Ask if they want noise with multi-start
            use_noise = safe_get_input(
                "\nAdd noise to help escape local minima during each run? (y/n, default = n): ",
                validation=lambda x: x in ['y', 'n', ''],
                error_msg="Enter y or n."
            )
            if use_noise == 'y':
                noise_amount = safe_get_input(
                    "Noise amount (e.g., 0.5 means ±50% random variation): ",
                    input_type=float,
                    validation=lambda x: x > 0,
                    error_msg="Noise amount must be positive."
                )
                noise_freq = safe_get_input(
                    "Inject noise every N iterations (e.g., 10): ",
                    input_type=int,
                    validation=lambda x: x > 0,
                    error_msg="Must be positive integer."
                )
            
        elif strategy_choice == '2':
            # Noise only (single run)
            multi_start = False
            num_starts = 1
            range_str = "-5,5"
            noise_amount = safe_get_input(
                "Noise amount (e.g., 0.5 means ±50% random variation): ",
                input_type=float,
                validation=lambda x: x > 0,
                error_msg="Noise amount must be positive."
            )
            noise_freq = safe_get_input(
                "Inject noise every N iterations (e.g., 10): ",
                input_type=int,
                validation=lambda x: x > 0,
                error_msg="Must be positive integer."
            )
            
        else:
            # Neither (default)
            multi_start = False
            num_starts = 1
            range_str = "-5,5"
            noise_amount = 0
            noise_freq = 10
        
        # Suggest initial learning rate
        suggested_lr, reason = suggest_learning_rate(func_str, start_values)
        print(f"\n    Initial Learning Rate Suggestion: {suggested_lr}")
        print(f"   Reason: {reason}")
        
        # Get learning rate with option to use suggested
        use_suggested = input(f"\nUse suggested initial learning rate {suggested_lr}? (y/n, or enter your own): ").lower()
        if use_suggested == 'y' or use_suggested == '':
            initial_lr = suggested_lr
            print(f"    Using suggested initial learning rate: {initial_lr}")
        else:
            initial_lr = safe_get_input(
                "Enter initial learning rate: ",
                input_type=float,
                validation=lambda x: x > 0,
                error_msg="Learning rate must be positive."
            )
        
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
            power = safe_get_input(
                "Enter power p (must be > 0.5, e.g., 0.75): ",
                input_type=float,
                validation=lambda x: x > 0.5,
                error_msg="Power must be greater than 0.5."
            )
        else:
            decay_type = 'inverse'
            power = 1.0
        
        max_iterations = safe_get_input(
            "Enter maximum number of iterations: ",
            input_type=int,
            validation=lambda x: x > 0,
            error_msg="Iterations must be a positive integer."
        )
        
        show_plots = True
        save_prefix = None
        
        # Print summary of what's being run
        print("\n" + "=" * 70)
        print("RUN SUMMARY")
        print("-" * 70)
        print(f"Function: {func_str}")
        print(f"Start: {start_values}")
        if multi_start:
            print(f"Multi-start: {num_starts} runs, range: {range_str}")
        if noise_amount > 0:
            print(f"Noise: {noise_amount} (every {noise_freq} iterations)")
        print(f"Learning Rate: {initial_lr}")
        print(f"Decay: {decay_type}")
        print(f"Iterations: {max_iterations}")
        print("-" * 70)
        
        # Run gradient descent
        return run_gradient_descent(
            func_str, start_values, initial_lr, decay_type, 
            power, max_iterations, show_plots, save_prefix,
            multi_start, num_starts, range_str,
            noise_amount, noise_freq
        )


if __name__ == "__main__":
    # Check if required libraries are installed
    try:
        import numpy
        import matplotlib
        import argparse
    except ImportError:
        print("\n      Required libraries not installed.")
        print("   Please install them with:")
        print("   pip install numpy matplotlib")
        print("\nThe gradient descent will still work, but visualizations won't.")
        print("-" * 70)
    
    # Parse command-line arguments
    args = parse_arguments()
    
    # Run with or without arguments
    try:
        gradient_descent_interactive(args)
    except KeyboardInterrupt:
        print("\n\n   Interrupted by user. Exiting...")
    except Exception as e:
        print(f"\n   Unexpected error: {e}")
        print("   Please check your input and try again.")
        if args and args.function:
            print("   Try running with --help for usage information.")
        exit(1)
