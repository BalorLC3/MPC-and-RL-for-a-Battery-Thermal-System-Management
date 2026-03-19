import jax.numpy as jnp
from control.jax.system.jax_ode_solver import rk4_step
import jax.tree_util
from control.baselines.config import SystemParameters

def _params_flatten(obj):
    keys = sorted(vars(obj).keys())
    children = [getattr(obj, k) for k in keys]
    aux_data = keys
    return children, aux_data

def _params_unflatten(aux_data, children):
    obj = SystemParameters.__new__(SystemParameters)
    for key, val in zip(aux_data, children):
        setattr(obj, key, val)
    return obj

jax.tree_util.register_pytree_node(
    SystemParameters,
    _params_flatten,
    _params_unflatten
)

class BatteryThermalSystem:
    def __init__(self, initial_state, params):
        self.params = params
        # Convert initial state to JAX array immediately
        self.state = jnp.array([
            initial_state['T_batt'],
            initial_state['T_clnt'],
            initial_state['soc']
        ])
        
        self.diagnostics = {}

    def step(self, controls, disturbances, dt):
        """
        Advances the simulation by one step using RK4 (JAX).
        """
        # Ensure inputs are JAX arrays
        controls_jax = jnp.array(controls)
        disturbances_jax = jnp.array(disturbances)
        
        # --- CALL THE JIT COMPILED SOLVER ---
        next_state, diag_vec = rk4_step(
            self.state, 
            controls_jax, 
            disturbances_jax, 
            self.params, 
            dt
        )
        
        self.state = next_state
        
        # Unpack diagnostics for plotting 
        # Order matches jax_ode_solver.py return

        self.diagnostics = {
            'P_cooling': diag_vec[0],
            'P_batt_total': diag_vec[1],
            'V_oc_pack': diag_vec[2],
            'I_batt': diag_vec[3],
            'Q_gen': diag_vec[4],
            'Q_cool': diag_vec[5],
            'm_clnt_dot': diag_vec[6],
            'P_comp_elec': diag_vec[7]
        }
        
        return self.state, self.diagnostics
