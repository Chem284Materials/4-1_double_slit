# Simulation of a Double Slit Experiment

In this problem, you will apply the time evolution operator to simulate the dynamics of a Gaussian wave packet that encounters a double slit.
The starter code (`animation.py`) initializes the system and creates an animation, but does not actually time evolve the system.

## Background

The time-dependent Schrodinger equation is

$$i\frac{\partial \psi}{\partial t} = \hat{H}\psi$$

where

$$\hat{H} = -\tfrac{1}{2}\nabla^2 + V(\mathbf{r})$$

We will compute $$\nabla^2\psi$$ numerically on a uniform grid.
Keep in mind the following method for approximating the derivative of a function for small values of $\Delta x$:

$$\frac{d f(x)}{d x} \approx \frac{1}{\Delta x} \[f(x + \Delta x) - f(x)\]$$

Similarly, we can numerically approximate the second derivative as:

$$\frac{d^2 f(x)}{d x^2} \approx \frac{1}{\Delta x} \[\frac{d f(x + \Delta x)}{d x} - \frac{d f(x - \Delta x)}{d x}\]$$

Substituting back in our approximation for the first derivative, we get:

$$\frac{d^2 f(x)}{d x^2} \approx \frac{1}{\Delta x} \[\frac{f(x + \Delta x) - f(x)}{\Delta x} - \frac{f(x) - f(x - \Delta x)}{\Delta x}\] = \frac{f(x + \Delta x) + f(x - \Delta x) - 2 f(x)}{\Delta x^2}$$

You can use this approximation when evaluating $$\nabla^2\psi$$.

