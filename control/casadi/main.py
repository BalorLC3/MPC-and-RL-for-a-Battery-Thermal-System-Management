"""
Run from repository root:

    python -m control.casadi.main
"""

import numpy as np
from control.casadi.system.sys_dynamics_casadi import BatteryThermalSystem, SystemParameters
from control.casadi.utils.setup import SimConfiguration, run_simulation
from control.casadi.controllers.thermostat import Thermostat
from control.casadi.controllers.mpc import DMPC, SMPC
from control.utils.plot_helper import show_results

if __name__ == "__main__":
    try: 
        driving_data = np.load('data/processed/driving_energy.npy', mmap_mode='r')
        velocity_data = np.load('data/processed/driving_velocity.npy', mmap_mode='r')
        print('Data loaded.')
    except:
        print("Change directory, files not found")
    dt = 1.0
    config = SimConfiguration(
        driving_data = driving_data,
        velocity_data = velocity_data,
        T_amb = 40.0,
        dt = dt
    )
    plot_config = 'horizontal'

    params = SystemParameters()
    init_state = {'T_batt': 30.0, 'T_clnt': 30.0, 'soc': 0.8}
    T_des = 33.0
    horizon = 10
    # ==========================================
    # SIMULACIÓN TERMOSTATO (BASELINE)
    # ==========================================
    print("\n--- Executing Thermostat ---")
    env_thermo = BatteryThermalSystem(init_state, params) # Instancia nueva
    ctrl_thermo = Thermostat()
    
    df_thermo = run_simulation(env_thermo, ctrl_thermo, config, verbose=0)
    show_results(df=df_thermo, controller_name='thermostat', config=plot_config)

    # ==========================================
    # SIMULACIÓN NMPC (DETERMINISTA)
    # ==========================================
    print("\n--- Executing DMPC ---")    
    ctrl_DMPC = DMPC(
        dt=dt, 
        T_des=T_des,
        horizon=horizon,     # Before horizon 5
        alpha=0.21, # 0.22
        avg_window=15
    )
    env_dmpc = BatteryThermalSystem(init_state, params)    
    df_dmpc = run_simulation(env_dmpc, ctrl_DMPC, config, verbose=0)
    show_results(df=df_dmpc, controller_name='dmpc', config=plot_config)

    # ==========================================
    # SIMULACIÓN SMPC (ESTOCASTICO)
    # ==========================================
    print("\n--- Executing SMPC ---")
    ctrl_SMPC = SMPC(
        driving_data,
        velocity_data,
        dt=dt,
        T_des=T_des,
        horizon=horizon,
        alpha=0.093, 
        n_clusters=4
    )

    env_smpc = BatteryThermalSystem(init_state, params)
    df_smpc = run_simulation(env_smpc, ctrl_SMPC, config, verbose=0)
    show_results(df=df_smpc, controller_name='smpc', config=plot_config)


