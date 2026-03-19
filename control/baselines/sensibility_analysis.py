'''
Here we we run both controllers from a single call, with a single
objective; see how sensible are both controllers to changes to key pa-
rameters

    python -m control.baselines.sensibility_analysis
'''

# Function which will interact with both metodologies
from control.utils.plot_helper import plot_sensitivity
from typing import Type

# MPC
import numpy as np
from control.casadi.system.sys_dynamics_casadi import BatteryThermalSystem, SystemParameters
from control.casadi.utils.setup import SimConfiguration, run_simulation
from control.casadi.controllers.thermostat import Thermostat
from control.casadi.controllers.mpc import DMPC, SMPC

# Reinforcement Learning
import jax
import jax.numpy as jnp
from control.jax.system.sys_dynamics_jax import SystemParameters
from control.jax.reinforcement_learning.sac import SBXActor
from control.jax.utils.setup import run_simulation, load_driving_cycle
from control.utils.plot_helper import show_results 
from control.jax.env.env_batt import ObservationConfig
import pickle


def modify_params(params: Type) -> None:
    ...