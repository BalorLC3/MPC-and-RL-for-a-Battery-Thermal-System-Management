# Battery Thermal Management (BTM)
This folder contains multiple `.py` files that I used for a Stochastic MPC (SMPC) controller. In SMPC the main idea is to treat external disturbances as a random variable (RV) which is bounded to a set $M$, with a computed density distribution.

Currently it only has a JAX and Casadi implementation.

## Key Features:
All control strategies were made for a high fidelity BTMS model, these strategies include the following:
- Stochastic Model Predictive Control formulation
- Deterministic Dynamic Programming
- Backward Dynamic Programming

TODO: Approximate DP.

## Context
- `sys_dynamics` - System dynamics which connect all other needed computations for the system
- `battery_models` - Battery thermo-electrical models

