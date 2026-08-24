# Gradient Descent Visualizer

A Python program that performs gradient descent on multi-dimensional functions with adaptive learning rates, 3D visualization, and advanced local minima escape strategies.

<div style="display: flex; overflow-x: auto; gap: 10px; padding: 10px 0;">
   <img width="1440" height="872" alt="gradient.py1" src="https://github.com/user-attachments/assets/babd8658-cf5a-4be2-949a-34045093eef1" />
   <img width="1440" height="900" alt="gradient.py2" src="https://github.com/user-attachments/assets/883bd464-b88f-4285-aec4-69f146e81082" />
   <img width="1440" height="900" alt="gradient.py3" src="https://github.com/user-attachments/assets/de7efd4e-b4ad-4695-b9d2-d2f00b5a778c" />
   <img width="1440" height="900" alt="gradient.py4" src="https://github.com/user-attachments/assets/57359959-cd95-4f67-82ec-bc6bca5833d5" />
   <img width="1440" height="900" alt="gradient.py5" src="https://github.com/user-attachments/assets/e84d299d-eb27-491a-b480-f83568ed3fd3" />
</div>

## What is Gradient Descent?

Gradient descent is an optimization algorithm used to find the minimum of a function. It works by:

1. Starting at some initial point
2. Calculating the slope (gradient) at that point
3. Taking a small step in the direction that decreases the function value most quickly
4. Repeating until the slope becomes nearly zero (indicating we've reached a minimum)

**Purpose**: Gradient descent is the backbone of machine learning. It's how neural networks learn by minimizing their error (loss) functions. By adjusting weights and biases step by step, models gradually improve their predictions.

## What This Program Does

This tool helps you understand how gradient descent works by:

- Running gradient descent on any function you type in
- Automatically suggesting a learning rate based on your function
- Showing each iteration step-by-step with variable values
- Visualizing the descent path in 3D (for 2-variable functions)
- Plotting convergence of all variables over time
- Displaying how the learning rate decays

## New Features

### Local Minima Escape Strategies

The program now includes two powerful strategies to help find the **global minimum** (not just a local one):

**1. Multi-start**: Runs gradient descent from multiple random starting points and keeps the best result. This explores different areas of the function simultaneously.

**2. Noise**: Adds random "jumps" to the learning rate at regular intervals, helping the algorithm bounce out of local minima and explore other valleys.

**3. Combined**: You can use both strategies together for maximum effectiveness!

### Command-Line Mode

Run the program non-interactively with command-line arguments for automation and scripting.

## Requirements

- Python 3.6 or higher
- numpy
- matplotlib

Install dependencies:

```bash
pip install numpy matplotlib
```

## How to Run

### Interactive Mode

1. Save the code as `gradient.py`

2. Run it:

```bash
python3 gradient.py
```

3. Follow the prompts:

   - Enter your function using `x1`, `x2`, `x3`, etc.
   - Enter starting values (comma-separated)
   - Choose a local minima escape strategy (Multi-start, Noise, or Neither)
   - Choose initial learning rate (or accept the suggestion)
   - Pick a decay schedule (recommended: inverse)
   - Set max iterations

### Command-Line Mode

Run the program with arguments for non-interactive use:

```bash
# Basic run
python3 gradient.py -f "x1**2 + x2**2" -s "3,4" -lr 0.1 -i 50

# Multi-start (10 random starts in range [-5,5])
python3 gradient.py -f "sin(x1)*cos(x2)" --multi 10 --range=-5,5

# With noise (adds random jumps to escape local minima)
python3 gradient.py -f "sin(x1)*cos(x2)" -s "1,1" --noise 0.5 --noise_freq 10

# Both multi-start AND noise
python3 gradient.py -f "sin(x1)*cos(x2)" --multi 10 --range=-5,5 --noise 0.3 --noise_freq 8
```

### Command-Line Options

| Flag | Description | Example |
|------|-------------|---------|
| `-f, --function` | Function to minimize | `"x1**2 + x2**2"` |
| `-s, --start` | Starting values (comma-separated) | `"3, 4"` |
| `-lr, --learning_rate` | Initial learning rate | `0.1` |
| `-d, --decay` | Decay schedule | `inverse`, `inverse_sqrt`, `inverse_power` |
| `-p, --power` | Power for inverse_power decay | `0.75` |
| `-i, --iterations` | Max iterations | `100` |
| `--multi` | Number of random starts | `10` |
| `--range` | Range for random starts | `--range=-5,5` |
| `--noise` | Noise amount | `0.5` |
| `--noise_freq` | Noise frequency | `10` |
| `--no-plots` | Disable plotting | (flag) |
| `--save` | Save plots with prefix | `"my_run"` |

## Examples

### Interactive Example

```
Enter your function: x1**2 + x2**2
Enter starting values: 3, 4
Use suggested learning rate? y
Choose decay schedule: 1
Max iterations: 50
```

### Multi-Start Example

```bash
# Find the global minimum of a wavy function with 10 starts
python3 gradient.py -f "sin(x1)*cos(x2)" --multi 10 --range=-5,5 --noise 0.3 --noise_freq 8
```

## Visualizations

- **3D Plot**: Shows the loss surface with the descent path (start=green, end=blue)
- **Convergence Plot**: Shows how each variable changes over iterations
- **LR Decay Plot**: Shows the learning rate decreasing over time

### Multi-Start Visualization

When using multi-start mode, the 3D plot shows:

- All trajectories in different colors
- The best trajectory highlighted in gold with a star marker
- Start points marked with dots

## Learning Rate Decay

The program uses diminishing learning rates that satisfy:

- ∑ηₙ = ∞ (enough total step length to reach the minimum)
- ∑ηₙ² < ∞ (steps get small enough to stop bouncing)

Three schedules are available:

1. **inverse**: ηₙ = η₀/(1 + η₀·n) — Recommended for most problems
2. **inverse_sqrt**: ηₙ = η₀/√n — Common in deep learning
3. **inverse_power**: ηₙ = η₀/n^p — Customizable decay rate

## Function Syntax

| Math | Type This |
|------|-----------|
| x₁² | `x1**2` |
| (x₁ - 2)² | `(x1 - 2)**2` |
| sin(x₁) + cos(x₂) | `sin(x1) + cos(x2)` |
| x₁² + x₂² + x₃² | `x1**2 + x2**2 + x3**2` |

## Tips for Finding Global Minima

1. **Use Multi-start**: For functions with many local minima, run with `--multi 20` or more starts
2. **Add Noise**: Combine multi-start with noise for even better exploration: `--multi 20 --noise 0.3`
3. **Wide Range**: Use a larger range like `--range=-10,10` to explore more of the function
4. **Check Plots**: The 3D visualization shows all trajectories, helping you understand the function's landscape

## Notes

- Works with any number of dimensions (2D gets 3D visualization)
- The inverse decay schedule is recommended for most problems
- If values explode, try a smaller initial learning rate
- The algorithm automatically detects the number of variables from your input
- All plots can be saved as PNG files using the `--save` flag
