# 1D Euler Shock Tube Solvers

This project numerically solves the 1D Euler equations for compressible hydrodynamics using:

- ✅ Lax-Friedrichs (1st order, diffusive)
- ✅ Lax-Wendroff (2nd order, oscillatory near shocks)
- ✅ HLL (robust Riemann solver)

Includes:
- Sod Shock Tube Problems A and B
- Contact discontinuity tracking using a passive tracer
- Feature extraction (shock, rarefaction, contact)
- Parallelization with OpenMP

## 🧮 Governing Equations

The conservative form of the Euler equations:

\[
\frac{\partial \mathbf{U}}{\partial t} + \frac{\partial \mathbf{F}(\mathbf{U})}{\partial x} = 0
\]

Where:

- \( \mathbf{U} = [\rho, \rho v, E]^\top \)
- \( \mathbf{F} = [\rho v, \rho v^2 + p, v(E + p)]^\top \)

Closed using the ideal gas law: \( p = (\gamma - 1)(E - \frac{1}{2}\rho v^2) \)

## 🧵 Compilation

```bash
make
