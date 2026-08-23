# Gradient Descent Visualizer

A Python program that performs gradient descent on multi-dimensional functions with adaptive learning rates and 3D visualization.

<img width="1440" height="872" alt="gradient.py1" src="https://github.com/user-attachments/assets/babd8658-cf5a-4be2-949a-34045093eef1" />

<img width="1440" height="900" alt="gradient.py2" src="https://github.com/user-attachments/assets/883bd464-b88f-4285-aec4-69f146e81082" />

<img width="1440" height="900" alt="gradient.py3" src="https://github.com/user-attachments/assets/de7efd4e-b4ad-4695-b9d2-d2f00b5a778c" />

<img width="1440" height="900" alt="gradient.py4" src="https://github.com/user-attachments/assets/57359959-cd95-4f67-82ec-bc6bca5833d5" />

<img width="1440" height="900" alt="gradient.py5" src="https://github.com/user-attachments/assets/e84d299d-eb27-491a-b480-f83568ed3fd3" />



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

## Requirements

- Python 3.6 or higher
- numpy
- matplotlib

Install dependencies:
```bash
pip install numpy matplotlib
```

## How to Run

1. Save the code as `gradient_descent.py`

2. Run it:
```bash
python3 gradient_descent.py
```

3. Follow the prompts:
   - Enter your function using `x1`, `x2`, `x3`, etc.
   - Enter starting values (comma-separated)
   - Choose initial learning rate (or accept the suggestion)
   - Pick a decay schedule (recommended: inverse)
   - Set max iterations

4. After the algorithm runs, choose whether to view plots

## Example

```
Enter your function: x1**2 + x2**2
Enter starting values: 3, 4
Use suggested learning rate? y
Choose decay schedule: 1
Max iterations: 50
```

The program will run gradient descent, show each step, then display:
- 3D surface plot with the descent path
- Convergence plot showing x1 and x2 over time
- Learning rate decay plot

## Function Syntax

| Math | Type This |
|------|-----------|
| x₁² | `x1**2` |
| (x₁ - 2)² | `(x1 - 2)**2` |
| sin(x₁) + cos(x₂) | `sin(x1) + cos(x2)` |
| x₁² + x₂² + x₃² | `x1**2 + x2**2 + x3**2` |

## Visualizations

- **3D Plot**: Shows the loss surface with the descent path (start=green, end=blue)
- **Convergence Plot**: Shows how each variable changes over iterations
- **LR Decay Plot**: Shows the learning rate decreasing over time

All plots can be saved as PNG files.

## Learning Rate Decay

The program uses diminishing learning rates that satisfy:
- ∑ηₙ = ∞ (enough total step length to reach the minimum)
- ∑ηₙ² < ∞ (steps get small enough to stop bouncing)

Three schedules are available:
1. **inverse**: ηₙ = η₀/(1 + η₀·n) — Recommended for most problems
2. **inverse_sqrt**: ηₙ = η₀/√n — Common in deep learning
3. **inverse_power**: ηₙ = η₀/n^p — Customizable decay rate

## Notes

- Works with any number of dimensions (2D gets 3D visualization)
- The inverse decay schedule is recommended for most problems
- If values explode, try a smaller initial learning rate
- The algorithm automatically detects the number of variables from your input
