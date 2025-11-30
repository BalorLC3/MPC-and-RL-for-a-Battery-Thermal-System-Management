# ===============================================================
# Part 1. Based on:
# Computationally Efficient Stochastic Model 
# Predictive Controller for Battery Thermal 
# Management of Electric Vehicle (2020).
#       Section III. Battery Cooling System Model
# ===============================================================
import numpy as np
import matplotlib.pyplot as plt
# --- Import additional data ---
entropy_data = np.load(r'C:\Users\super\Desktop\Vasudeva\Manifold\engineering\predictive-control\stochastic-mpc\entropy_data.npz') 
battery_parameters = np.load(r'C:\Users\super\Desktop\Vasudeva\Manifold\engineering\predictive-control\stochastic-mpc\battery_parameters.npz')

# --- Cooling System Parameters (Table 1)---
rho_rfg  = 27.8         # [kg/m^3] Density of refrigerant
rho_clnt = 1069.5       # [kh/m^3] Density of coolant
m_batt   = 40           # [kg]     Battery thermal mass 
C_rfg    = 1117         # [J/kg/C] Specific heat capacity of rfg.
C_clnt   = 3330         # [J/kg/C] Specific heat capacity of coolant
C_batt   = 1350         # [J/kg/C] Battery specific heat capacity
V_comp   = 33e10-6      # [m^3/rv] Displacement volume of compressor
V_pump   = 33e10-6      # [m^3/rv] Displacement volume of pump
h_eva    = 1000         # [W/m^2/C]Heat transfer coefficient between evaporator and refrigerant/coolant 
h_batt   = 300          # [W/m^2/C]Heat transfer coefficient between the battery and the coolant
A_eva    = 0.3          # [m^2]    Heat transfer sectional area between evaporator and refrigerant/coolant
A_batt   = 1            # [m^2]    Heat transfer sectional area between the battery and the coolant 
PR       = 5            # [ ]      Compression ratio of the compressor
h_cout   = 284.3        # [kJ/kg]  Enthalpy at the outlet of compressor
h_evaout = 250.9        # [kJ/kg]  Enthalpy at the outlet of evaporator
h_condout= 131.7        # [kJ/kg]  Enthalpy at the outlet of condenser
p_compout= 1500         # [kpa]    Pressure at the outlet of compressor
p_compin = 300          # [kpa]    Pressure at the outlet of compressor
p_condout= 1500         # [kpa]    Pressure at the outlet of condenser

# ===============================================================
# Part2. Parameters extracted from
# Entropy Profiles for Li-Ion Batteries—Effects of Chemistries and Degradation (2025).
#       From supplement material: https://www.mdpi.com/article/10.3390/e27040364/s1.
# ===============================================================
soc_points = entropy_data['soc_points']
dvdt_points = entropy_data['dvdt_points']

# --- System governing equation ---
def get_dvdt(soc):
    '''This function yields an interpolation for the required SoC'''
    return np.interp(soc, soc_points, dvdt_points)

def opencircuitvoltage(SOC, p0=3.637, 
             p1=-0.3091, 
             p2=7.033e-5, 
             alpha1=-0.0005747, 
             alpha2=-0.1366):
    return p0*np.exp(alpha1*SOC) + p1*np.exp(alpha2*SOC)+ p2*SOC**2

print(opencircuitvoltage(50))

print(battery_parameters)