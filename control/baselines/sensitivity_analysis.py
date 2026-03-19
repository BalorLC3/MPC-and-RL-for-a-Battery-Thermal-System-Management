'''
Here we run both controllers from a single call, with a single
objective; see how sensible are both controllers to changes to key
parameters

    python -m control.baselines.sensitivity_analysis
'''

import numpy as np
import jax.numpy as jnp
import pickle

from control.baselines.config import SystemParameters
from control.utils.plot_helper import show_sensitivity

# CasADi / MPC
from control.casadi.system.sys_dynamics_casadi import BatteryThermalSystem
from control.casadi.utils.setup import SimConfiguration, run_simulation as run_simulation_ca
from control.casadi.controllers.mpc import DMPC

# JAX / SAC
from control.jax.reinforcement_learning.sac import SBXActor
from control.jax.utils.setup import run_simulation as run_simulation_jax, load_driving_cycle
from control.jax.env.env_batt import ObservationConfig
import jax


# ---------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------

def make_varied_params(variance: float) -> SystemParameters:
    """
    Returns a fresh SystemParameters scaled by (1 + variance).
    """
    params = SystemParameters()
    params.r_int_scale       *= (1 + variance)
    params.h_batt            *= (1 + variance)
    params.comp_max_isen_eff *= (1 + variance)
    return params


def run_dmpc(params, config, init_state, dt, T_des, horizon):
    ctrl = DMPC(dt=dt, T_des=T_des, horizon=horizon, alpha=0.21, avg_window=15)
    env  = BatteryThermalSystem(init_state, params)
    return run_simulation_ca(env, ctrl, config, verbose=0)


def run_sac(params, dist, init_state, dt, controller_name, horizon):
    with open(f"control/jax/results/{controller_name}/actor_weights.pkl", "rb") as f:
        params_nn = pickle.load(f)

    actor      = SBXActor(n_actions=2)
    obs_config = ObservationConfig(horizon=horizon)

    def get_obs(state, disturbance, preview):
        raw   = jnp.concatenate([state, disturbance, preview])
        mean  = jnp.concatenate([obs_config.obs_mean,  jnp.full((obs_config.horizon,), 10000.0)])
        scale = jnp.concatenate([obs_config.obs_scale, jnp.full((obs_config.horizon,), 10000.0)])
        return (raw - mean) / scale

    def controller_fn(state, carry, k, params_sys):
        d_curr  = dist[k]
        preview = jnp.zeros((obs_config.horizon,))
        if obs_config.horizon > 0:
            preview = jax.lax.dynamic_slice(
                dist, (k + 1, 0), (obs_config.horizon, 1)
            ).reshape(-1)
        obs      = get_obs(state, d_curr, preview)
        action   = jnp.tanh(actor.apply(params_nn, obs))
        controls = (action + 1.0) * 5000.0
        return controls, carry

    return run_simulation_jax(init_state, controller_fn, dist, params, dt)


# ---------------------------------------------------------------
# Main
# ---------------------------------------------------------------

if __name__ == "__main__":

    dt        = 1.0
    T_des     = 33.0
    horizon   = 10
    sac_name  = "sac_h0"

    init_state_ca  = {'T_batt': 30.0, 'T_clnt': 30.0, 'soc': 0.8}
    init_state_jax = jnp.array([30.0, 30.0, 0.8])

    driving_data  = np.load('data/processed/driving_energy.npy',   mmap_mode='r')
    velocity_data = np.load('data/processed/driving_velocity.npy', mmap_mode='r')
    dist          = load_driving_cycle()

    config = SimConfiguration(
        driving_data=driving_data,
        velocity_data=velocity_data,
        T_amb=40.0,
        dt=dt,
    )

    variances  = {'minus': -0.2, 'base': 0.0, 'plus': 0.2}
    df_dmpc    = {}
    hist_sac   = {}

    for key, variance in variances.items():
        print(f"\n |  >  variance = {variance:+.0%}  <  |")
        params = make_varied_params(variance)

        print("  DMPC...")
        df_dmpc[key]  = run_dmpc(params, config, init_state_ca, dt, T_des, horizon)

        print(f"  SAC ({sac_name})...")
        hist_sac[key] = run_sac(params, dist, init_state_jax, dt, sac_name, horizon=0)

    # --- Plot ---
    show_sensitivity(
        controller_name='dmpc_sensitivity',
        config='horizontal',
        dt=dt,
        df_minus=df_dmpc['minus'],
        df_base=df_dmpc['base'],
        df_plus=df_dmpc['plus'],
    )

    show_sensitivity(
        controller_name='sac_sensitivity',
        config='horizontal',
        dt=dt,
        hist_minus=hist_sac['minus'],
        hist_base=hist_sac['base'],
        hist_plus=hist_sac['plus'],
    )