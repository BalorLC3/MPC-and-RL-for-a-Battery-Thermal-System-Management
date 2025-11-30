import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# ===============================================================
# SCRIPT DE PRE-PROCESAMIENTO DEL CICLO DE CONDUCCIÓN
#
# Propósito: Leer un archivo de texto de un ciclo de conducción (ej. UDDS),
# convertirlo en un perfil de potencia de manejo (P_driv), y guardarlo
# como un archivo .npy para ser usado como perturbación en la simulación.
# ===============================================================

# --- 1. CONFIGURACIÓN Y PARÁMETROS DEL VEHÍCULO ---

# Parámetros para la conversión de unidades
MPH_TO_MPS = 0.44704  # 1 milla por hora = 0.44704 metros por segundo

# Parámetros del Vehículo (valores típicos para un sedán eléctrico)
M_VEHICLE = 1900.0     # Masa del vehículo [kg]
G_ACCEL = 9.81         # Aceleración de la gravedad [m/s^2]
C_RR = 0.02            # Coeficiente de resistencia a la rodadura (adimensional)
RHO_AIR = 1.2          # Densidad del aire [kg/m^3]
A_FRONTAL = 2.8        # Área frontal del vehículo [m^2]
C_DRAG = 0.35          # Coeficiente de arrastre aerodinámico (adimensional)
DRIVETRAIN_EFF = 0.80  # Eficiencia del tren motriz (motor + transmisión)
REGEN_EFF = 0.65       # Eficiencia del frenado regenerativo

# Rutas de los archivos
input_file_path = r'C:\Users\super\Desktop\Supernatural\TESIS\thermal-management\UDDS.txt' 
# Dónde guardar el archivo .npy procesado
output_file_path = r'C:\Users\super\Desktop\Vasudeva\Manifold\engineering\predictive-control\stochastic-mpc\driving_energy.npy'


# --- 2. FUNCIÓN DE DINÁMICA VEHICULAR ---

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
    df = pd.read_csv(input_file_path, sep='\s+', skiprows=1, names=['time_s', 'speed_mph'])
except FileNotFoundError:
    print(f"Error: No se encontró el archivo del ciclo de conducción en {input_file_path}")
    exit()

# Conversión de unidades
df['speed_mps'] = df['speed_mph'] * MPH_TO_MPS

dt = df['time_s'].diff().iloc[1] 
df['accel_mps2'] = df['speed_mps'].diff() / dt
# Rellenar el primer valor NaN con 0
df['accel_mps2'].fillna(0, inplace=True)

# Aplicar la función de dinámica vehicular para calcular P_driv para cada fila
df['P_driv_W'] = df.apply(lambda row: calculate_driving_power(row['speed_mps'], row['accel_mps2']), axis=1)

# Extraer el perfil de potencia como un array de NumPy
p_driv_profile = df['P_driv_W'].values

# Guardar el perfil para usarlo en la simulación
np.save(output_file_path, p_driv_profile)

print(f"Perfil de potencia procesado y guardado exitosamente en: {output_file_path}")
print(f"Número de puntos de datos: {len(p_driv_profile)}")


# --- 4. VISUALIZACIÓN PARA VERIFICACIÓN ---

fig, axs = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

# Gráfico 1: Perfil de Velocidad
axs[0].plot(df['time_s'], df['speed_mph'], 'b-')
axs[0].set_title('Ciclo de Conducción UDDS - Perfil de Velocidad')
axs[0].set_ylabel('Velocidad (mph)')
axs[0].grid(True)

# Gráfico 2: Perfil de Aceleración
axs[1].plot(df['time_s'], df['accel_mps2'], 'g-')
axs[1].set_title('Perfil de Aceleración Calculado')
axs[1].set_ylabel('Aceleración (m/s²)')
axs[1].grid(True)

# Gráfico 3: Perfil de Potencia Resultante (P_driv)
axs[2].plot(df['time_s'], df['P_driv_W'] / 1000, 'r-') # en kW para mejor escala
axs[2].set_title('Perfil de Potencia de Manejo (P_driv) Resultante')
axs[2].set_ylabel('Potencia (kW)')
axs[2].set_xlabel('Tiempo (s)')
axs[2].grid(True)
axs[2].axhline(0, color='k', linestyle='--', linewidth=0.8)
axs[2].text(500, 25, 'Propulsión', color='red')
axs[2].text(500, -15, 'Frenado Regenerativo', color='red')

plt.tight_layout()
plt.show()