import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path
# ===============================================================
# PREPROCESSING OF DRIVING FILES
# ===============================================================
plt.rcParams.update({
    "text.usetex": False,
    "font.family": "sans-serif",
    "font.sans-serif": ["Microsoft YaHei"], # Change this to tex True & unicode to True
    "axes.labelsize": 12,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "axes.spines.top": True,
    "axes.spines.right": True,
    "axes.spines.left": True,
    "axes.spines.bottom": True,
    "figure.figsize": (6.0, 10.0),
    "lines.linewidth": 1.4,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linestyle": "--",
    "savefig.dpi": 300
})

# Unit conversion
MPH_TO_MPS = 0.44704   

# Vehicle parameters (typical of a electrical sedan)
M_VEHICLE = 1850.0     # Vehicle mass [kg]
G_ACCEL = 9.81         # Gravity acceleration [m/s^2]
C_RR = 0.02            # Coefficient of rolling resistance (adimensional)
RHO_AIR = 1.2          # Air density [kg/m^3]
A_FRONTAL = 2.8        # Área frontal del vehículo [m^2]
C_DRAG = 0.35          # Coeficiente de arrastre aerodinámico (adimensional)
DRIVETRAIN_EFF = 0.80  # Eficiencia del tren motriz (motor + transmisión)
REGEN_EFF = 0.65       # Eficiencia del frenado regenerativo

filename = 'driving_data/UDDS.txt'
output_dir = Path.cwd() / filename
power_output_path = os.path.join(output_dir, 'driving_energy.npy')
velocity_output_path = os.path.join(output_dir, 'driving_velocity.npy')
time_output_path = os.path.join(output_dir, 'driving_time.npy')


def calculate_driving_power(v, a):
    """
    Calcula la potencia eléctrica requerida por la batería (P_driv) para
    una velocidad 'v' y aceleración 'a' dadas.
    
    Args:
        v (float): Velocidad del vehículo [m/s].
        a (float): Aceleración del vehículo [m/s^2].
        
    Returns:
        float: Potencia requerida [W]. Positiva para propulsión, negativa para regeneración.
    """
    # Fuerza de resistencia a la rodadura
    f_roll = M_VEHICLE * G_ACCEL * C_RR
    
    # Fuerza de arrastre aerodinámico
    f_aero = 0.5 * RHO_AIR * A_FRONTAL * C_DRAG * v**2
    
    # Fuerza de inercia (para acelerar)
    f_accel = M_VEHICLE * a
    
    # Potencia total requerida en las ruedas
    p_wheels = (f_roll + f_aero + f_accel) * v
    
    # Contabiliza las eficiencias del tren motriz
    if p_wheels >= 0:
        # Propulsión: la batería debe entregar más potencia de la que llega a las ruedas
        p_driv = p_wheels / DRIVETRAIN_EFF
    else:
        # Frenado regenerativo: solo una parte de la energía de frenado vuelve a la batería
        p_driv = p_wheels * REGEN_EFF
        
    return p_driv


# --- 3. PROCESAMIENTO DEL ARCHIVO DEL CICLO DE CONDUCCIÓN ---

try:
    df = pd.read_csv(filename, sep='\s+', skiprows=1, names=['time_s', 'speed_mph'])
except FileNotFoundError:
    print(f"Error: No se encontró el archivo del ciclo de conducción en {filename}")
    exit()

# Conversión de unidades
df['speed_mps'] = df['speed_mph'] * MPH_TO_MPS

dt = df['time_s'].diff().iloc[1] 
df['accel_mps2'] = df['speed_mps'].diff() / dt
# Rellenar el primer valor NaN con 0
df['accel_mps2'].fillna(0, inplace=True)

# Aplicar la función de dinámica vehicular para calcular P_driv para cada fila
df['P_driv_W'] = df.apply(lambda row: calculate_driving_power(row['speed_mps'], row['accel_mps2']), axis=1)

# Extraer los perfiles como arrays de NumPy
p_driv_profile = df['P_driv_W'].values
velocity_profile = df['speed_mps'].values
time_profile = df['time_s'].values

# Duplicate driving cycle (this is the modification)
p_driv_profile_duplicated = np.concatenate([p_driv_profile, p_driv_profile])
velocity_profile_duplicated = np.concatenate([velocity_profile, velocity_profile])

total_original_time = time_profile[-1] + dt  # Total time of original cycle
time_duplicated = np.concatenate([time_profile, time_profile + total_original_time])

# Save profiles
np.save(power_output_path, p_driv_profile_duplicated)
np.save(velocity_output_path, velocity_profile_duplicated)



fig, axs = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

axs[0].plot(time_duplicated, p_driv_profile_duplicated / 1000, 'r-') 
axs[0].set_ylabel('Potencia (kW)')
axs[0].grid(True)
axs[0].axhline(0, color='black', lw='1.5', ls='--')
axs[0].text(500, 30, 'Propulsión', color='black')
axs[0].text(500, -20, 'Frenado Regenerativo', color='black')

axs[1].plot(time_duplicated, velocity_profile_duplicated, 'b-')
axs[1].set_ylabel('Velocidad (m/s)')
axs[1].grid(True)
axs[1].set_xlabel('Tiempo (s)')



plt.tight_layout()

plt.show()

print("\n--- ESTADÍSTICAS COMPARATIVAS ---")
print(f"Potencia máxima (propulsión): {p_driv_profile.max()/1000:.2f} kW")
print(f"Potencia mínima (regeneración): {p_driv_profile.min()/1000:.2f} kW")
print(f"Velocidad máxima: {velocity_profile.max():.2f} m/s ({velocity_profile.max()/MPH_TO_MPS:.2f} mph)")
print(f"Velocidad mínima: {velocity_profile.min():.2f} m/s ({velocity_profile.min()/MPH_TO_MPS:.2f} mph)")
print(f"Energía neta por ciclo: {np.trapz(p_driv_profile, dx=dt)/3600000:.4f} kWh")
print(f"Energía total para dos ciclos: {np.trapz(p_driv_profile_duplicated, dx=dt)/3600000:.4f} kWh")