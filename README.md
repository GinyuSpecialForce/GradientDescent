# Gradient Descent Visualizer

A Python program that performs unconstrained projected gradient descent on multi-dimensional functions with adaptive learning rates, 3D visualization, color gradient trajectories, animated GIFs, and advanced local minima escape strategies.

<div style="display: flex; overflow-x: auto; gap: 10px; padding: 10px 0;">
  <img width="1920" height="1200" alt="readme" src="https://github.com/user-attachments/assets/4d7aaebd-4487-496c-9c41-d0bf96c743e4" />
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
- Visualizing the descent path in 3D with color-coded trajectories
- Creating animated GIFs of the descent process
- Plotting convergence of all variables over time
- Displaying how the learning rate decays

## Features

### Local Minima Escape Strategies

The program includes two powerful strategies to help find the **global minimum** (not just a local one):

**1. Multi-start**: Runs gradient descent from multiple random starting points and keeps the best result. This explores different areas of the function simultaneously.

**2. Noise**: Adds random "jumps" to the learning rate at regular intervals, helping the algorithm bounce out of local minima and explore other valleys.

**3. Combined**: Use both strategies together for maximum effectiveness.

### Color Gradient Visualization

The descent path is displayed with a color gradient (purple to yellow) showing iteration progress. This helps visualize:
- Where the algorithm started (green)
- Where it ended (red)
- The speed of convergence (color changes indicate progress)

### Animated GIFs

Create animated GIFs of the descent process showing:
- The path evolving step by step
- Current iteration number displayed
- Progress indicator

### Clamping

When using multi-start with a specified range, the algorithm will clamp values that try to move outside the range. This keeps the search within the bounds you specify and helps prevent the algorithm from wandering too far. The program tracks clamping events and displays them in the summary.

### Command-Line Mode

Run the program non-interactively with command-line arguments for automation and scripting.

## Requirements

- Python 3.6 or higher
- numpy
- matplotlib
- pillow (for GIF creation)

Install dependencies:

```bash
pip install numpy matplotlib pillow
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
   - Choose whether to create an animated GIF
   - Choose initial learning rate (or accept the suggestion)
   - Pick a decay schedule (recommended: inverse)
   - Set max iterations

### Command-Line Mode

Run the program with arguments for non-interactive use:

```bash
# Basic run with color gradient
python3 gradient.py -f "x1**2 + x2**2" -s "3,4" -lr 0.1 -i 50

# Multi-start (10 random starts in range [-5,5])
python3 gradient.py -f "sin(x1)*cos(x2)" --multi 10 --range=-5,5

# With noise (adds random jumps to escape local minima)
python3 gradient.py -f "sin(x1)*cos(x2)" -s "1,1" --noise 0.5 --noise_freq 10

# Both multi-start AND noise
python3 gradient.py -f "sin(x1)*cos(x2)" --multi 10 --range=-5,5 --noise 0.3 --noise_freq 8

# Create an animated GIF of the descent
python3 gradient.py -f "x1**2 + x2**2" -s "3,4" -lr 0.1 -i 50 --animate --save my_run
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
| `--animate` | Create an animated GIF | (flag) |

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
python3 gradient.py -f "sin(x1)*cos(x2)" --multi 10 --range=-5,5 --noise 0.3 --noise_freq 8 --animate --save my_run
```

## Visualizations

### Single Run
- **3D Plot**: Shows the loss surface with the descent path (start=green, end=blue)
- **2D Color Gradient Plot**: Shows the descent path color-coded by iteration progress
- **Convergence Plot**: Shows how each variable changes over iterations
- **LR Decay Plot**: Shows the learning rate decreasing over time
- **Animation**: Optional animated GIF showing the step-by-step descent

### Multi-Start
- **Best Run Convergence**: Shows the convergence of the best run
- **Best Run LR Decay**: Learning rate decay for the best run
- **Best Non-Clamping Run**: If clamping occurred, shows the best run that stayed within bounds
- **3D Trajectory Plot**: Shows all trajectories in different colors, with the best path in gold
- **2D Color Gradient Plot**: Shows the best run with color-coded iteration progress

All plots can be saved as PNG files using the `--save` flag. Animations are saved as GIFs.

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
5. **Clamping**: If you see clamping events in the summary, try increasing the range or using a smaller learning rate
6. **Animation**: Use `--animate` to watch the descent step by step, which helps understand how the algorithm works

## Notes

- Works with any number of dimensions (2D gets 3D visualization and color gradient plots)
- The inverse decay schedule is recommended for most problems
- If values explode, try a smaller initial learning rate
- The algorithm automatically detects the number of variables from your input
- All plots can be saved as PNG files using the `--save` flag
- Animations can be saved as GIFs using `--save` with `--animate`
- Clamping events are tracked and displayed when using multi-start with a range
- The best non-clamping run is shown when clamping occurs
- Color gradient trajectories show iteration progress from purple (start) to yellow (end)
