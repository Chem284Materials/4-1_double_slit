# CUDA Simulation of a Double Slit Experiment

In this problem, you will apply the time evolution operator to simulate the dynamics of a Gaussian wave packet that encounters a double slit.
The starter code (`animation.py`) initializes the system and creates an animation, but does not actually time evolve the system.

## Background

The time-dependent Schrodinger equation is

$$i\frac{\partial \psi(\vec{x}, t)}{\partial t} = \hat{H}\psi(\vec{x}, t)$$

where

$$\hat{H} = -\tfrac{1}{2}\nabla^2 + V(\mathbf{r})$$

If we have a wavefunction at any point in time, we can use the time evolution operator, $$e^{-i \hat{H} t/ \hbar}$$, to obtain it at some future time $t$:

$$\psi(\vec{x}, t) = e^{-i \hat{H} t/ \hbar}\psi(\vec{x}, 0)$$

It might not be particularly obvious what $e^{-i \hat{H} t/ \hbar}\psi(\vec{x}, t)$ actually *means*, though.
How does an operator like $\hat{H}$ or $\nabla^2$ work when it is in an exponent?
A common way to handle the time evolution operator is to expand it in a Taylor Series:

$$e^{-i \hat{H} t/ \hbar} = 1 - i \hat{H} \Delta t - \frac{1}{2} \hat{H}^2 (\Delta t)^2 + ...$$

For sufficiently small values of $\Delta t$, we can just use the first two terms ($1 - i \hat{H} \Delta t$).
This gives us a straightforward way to perform simulations of a wavefunction over some period of time.
Each timestep, the new wavefunction is:

$$\psi(\vec{x}, t+\Delta t) = (1 - i \hat{H} \Delta t) \psi(\vec{x}, t)$$

You will use this approximation to perform a dynamics simulation of an electron that is approaching a double slit.



## Your Task

If you run the code in this repository (`animation.py`), you will find that it creates file, `animation.gif`, that should show the time evolution of an electron's probability distribution.
Currently, the code does not perform any time evolution; you must implement the time-evolution functionality described above, **using CUDA**.

Set $\hbar$ and the mass of the particle to 1 in your code.

Note that a potential energy surface (a hard wall with two slits) is created by the starting code.
The time evolution you implement should incorporate this potential into $\hat{H}$.

After getting everything working, upload your `animation.gif` file to this repository.

### Numerical Derivatives

We will compute $$\nabla^2\psi$$ numerically on a uniform grid.
Keep in mind the following method for approximating the derivative of a function for small values of $\Delta x$:

$$\frac{d f(x)}{d x} \approx \frac{1}{\Delta x} \[f(x + \Delta x) - f(x)\]$$

Similarly, we can numerically approximate the second derivative as:

$$\frac{d^2 f(x)}{d x^2} \approx \frac{1}{\Delta x} \[\frac{d f(x + \Delta x)}{d x} - \frac{d f(x - \Delta x)}{d x}\]$$

Substituting back in our approximation for the first derivative, we get:

$$\frac{d^2 f(x)}{d x^2} \approx \frac{1}{\Delta x} \[\frac{f(x + \Delta x) - f(x)}{\Delta x} - \frac{f(x) - f(x - \Delta x)}{\Delta x}\] = \frac{f(x + \Delta x) + f(x - \Delta x) - 2 f(x)}{\Delta x^2}$$

You can use this approximation when evaluating $$\nabla^2\psi$$.

### Symplectic Time Integration

There is one other subtle nuance that is very important to keep in mind while solving this problem.
If you don't use a [symplectic integrator](https://en.wikipedia.org/wiki/Symplectic_integrator), **your code will exhibit massive numerical instabilities**.
Symplectic integrators are a fairly niche subject, and it isn't necessary for you to understand the topic in detail.
The important thing to understand is that **naive implementations that update both the real and imaginary part of the wavefunction simultaneously will not be numerically stable**.

The simplest way to implement symplectic integration in this case is to first update the real part of the wavefunction, and then update the imaginary part.
If our wavefunction has a real part called $\psi_{re}$ and an imaginary part called $\psi_{im}$, so that

$$\psi = \psi_{re} + i \psi_{im}$$

then what we can do is first update the real component of the wavefunction:

$$\psi_{re}(\vec{x},t+\Delta t) = \psi_{re}(\vec{x},t) +  \hat{H} \psi_{im}(\vec{x},t) \Delta t$$

and only **after finishing the update of the real component**, we can then begin calculating and applying the update to the imaginary component:

$$\psi_{im}(\vec{x},t+\Delta t) = \psi_{im}(\vec{x},t) - \hat{H} \psi_{re}(\vec{x},t + \Delta t) \Delta t$$
