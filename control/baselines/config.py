import numpy as np

class SystemParameters:
    def __init__(self):
        # --- Thermodynamics ---
        self.rho_rfg   = 27.8
        self.rho_clnt  = 1069.5
        self.C_rfg     = 1117.0
        self.C_clnt    = 3330.0
        self.V_comp    = 33e-6
        self.V_pump    = 33e-6
        self.h_eva     = 1000.0
        self.A_eva     = 0.3
        self.h_batt    = 300.0     
        self.A_batt    = 1.0
        self.PR        = 5.0
        self.h_cout_kJ    = 284.3
        self.h_evaout_kJ  = 250.9

        # --- Battery ---
        self.m_batt     = 40.0
        self.C_batt     = 1350.0
        self.N_series   = 96.0
        self.N_parallel = 1.0

        self.m_clnt_total = 2.0 * self.rho_clnt / 1000

        # Compressor 
        self.comp_max_speed_rpm      = 10000.0
        self.comp_nominal_speed_rpm  = 6000.0
        self.comp_max_vol_eff        = 0.98
        self.comp_max_isen_eff       = 0.80  

        # Pump
        self.pump_max_speed_rpm  = 8000.0
        self.pump_max_vol_eff    = 0.98
        self.pump_pressure_coeff = 3300.0

        # Engine 
        self.motor_max_eff          = 0.92
        self.motor_nominal_speed_rpm = 5000.0

        # Resistance
        self.r_int_scale = 1.0 # For sensibility