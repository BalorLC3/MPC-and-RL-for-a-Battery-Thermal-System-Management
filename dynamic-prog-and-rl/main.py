# All time modules
import numpy as np
import jax
import jax.numpy as jnp
# Own modules
from system.sys_dynamics_jax import SystemParameters
from controllers.dynamic_programming import run_dp_offline, make_dp_controller_fn
from controllers.thermostat import thermostat_logic_jax
from controllers.sac import SBXActor
from utils.setup import run_simulation
from utils.plot_funs import show_results
from src.env_batt import ObservationConfig
import pickle

# Mean: [T_b, T_c, SOC, P_driv, T_amb]
obs_config = ObservationConfig()

def get_obs(state, disturbance, preview):
    raw = jnp.concatenate([state, disturbance, preview])

    mean = jnp.concatenate([
        obs_config.obs_mean,
        jnp.full((obs_config.horizon,), 10000.0)
    ])
    scale = jnp.concatenate([
        obs_config.obs_scale,
        jnp.full((obs_config.horizon,), 10000.0)
    ])
    return (raw - mean) / scale

def inference_fn(state, carry, k, params):
    """Model - free, it does not interact with the parameters of the system but with the actions"""
    d_curr = dist[k]
    
    start_indices = (k + 1, 0)
    slice_sizes = (obs_config.horizon, 1)

    preview_slice = jax.lax.dynamic_slice(dist, start_indices, slice_sizes)

    preview = preview_slice.reshape(-1)

    obs = get_obs(state, d_curr, preview)
    
    mean = actor.apply(params_nn, obs)
    action = jnp.tanh(mean) # [-1, 1]
    
    controls = (action + 1.0) * 5000.0 # [0, 10_000]ssss
    
    return controls, carry


if __name__ == "__main__":
    # ===============================================================
    # SETUP (N, data, disturbances and parameters)
    # ===============================================================
    try:
        raw = np.load('data/driving_energy.npy', mmap_mode='r')
        data = jnp.array(raw)
        print("Driving cycle loaded.")
    except:
        print("Loading of driving cycle failed, using fallback data.")
        data = jnp.abs(jnp.sin(jnp.arange(600)/50.0)) * 20000.0
    N = len(data)
    dist = jnp.hstack([data.reshape(-1,1), jnp.full((N,1), 40.0)])
    params = SystemParameters()
    # ===============================================================
    # CONTROLLER 
    # ===============================================================
    controller_name = "sac_horizon"

    if controller_name == "dp":
        policy_cube = run_dp_offline(dist, params, alpha=100.0)
        controller_func = make_dp_controller_fn(policy_cube)
    elif controller_name == "thermostat":
        controller_func = thermostat_logic_jax
    elif controller_name == "sac" or "sac_horizon":
            print("Executing learned model SAC:", controller_name)
            with open(f'results/{controller_name}_actor_weights.pkl', 'rb') as f:
                params_nn = pickle.load(f)
                
            actor = SBXActor(n_actions=2)
            
            controller_func = inference_fn

    # ===============================================================
    # SHARED FUNCTIONALITIES (initial state and always must yield history)
    # ===============================================================
    init_state = jnp.array([30.0, 30.0, 0.8])
    history = run_simulation(init_state, controller_func, dist, params, 1.0)
    # ===============================================================
    # RESULTS 
    # ===============================================================
    states_hist = np.array(history['state'])
    ctrl_hist = np.array(history['controls'])
    diag_hist = np.array(history['diagnostics'])
    
    show_results(controller_name, N, states_hist, ctrl_hist, diag_hist)
