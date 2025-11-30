import numpy as np
import os
import matplotlib.pyplot as plt

# ===============================================================
# MÓDULO DE MODELOS DE BATERÍA
#
# Este script contiene todos los modelos de parámetros de la batería,
# basados en los datos de Baccouche et al. (2017).
# Las funciones aquí definidas pueden ser importadas directamente
# en el script de simulación principal.
# ===============================================================

# --- PARÁMETROS GLOBALES Y DATOS TRANSCRITOS ---

# --- Datos del Modelo OCV (de la Tabla 7) ---
ocv_temperatures = np.array([5, 15, 25, 45])
ocv_params_charge = np.array([
    [3.734, -0.2756, 8.013e-5, -0.001155, -0.06674],
    [3.629, -0.3191, 6.952e-5, -0.0005411, -0.1493],
    [3.637, -0.3091, 7.033e-5, -0.0005477, -0.1366],
    [3.584, -0.4868, 6.212e-5, -0.0001919, -0.2181],
])
ocv_params_discharge = np.array([
    [3.804, -0.3487, 8.838e-5, -0.001618, -0.04724],
    [3.599, -0.2933, 6.9e-5,   -0.0004687, -0.1434],
    [3.604, -0.2803, 6.871e-5, -0.0004523, -0.1341],
    [3.55,  -0.4573, 6.02e-5,  -6.241e-5,  -0.221],
])

# --- Datos del Modelo de Resistencia y Capacidad (de la Tabla 2) ---
res_temperatures = np.array([5, 15, 25, 45])
# Capacidad Nominal [Ah]
cnom_table = np.array([17.17, 19.24, 20.0, 21.6])
# Resistencia Óhmica R_i (R0) [Ohms]
r0_table = np.array([0.007, 0.0047, 0.003, 0.0019])
# Resistencia de Difusión R_df (R1) [Ohms]
r1_table = np.array([0.0042, 0.0018, 0.00065, 0.00054])

soc_table_res = np.array([0.0, 0.1, 0.2, 0.4, 0.6, 0.8, 0.9, 1.0])
resistance_multiplier = np.array([1.8, 1.3, 1.1, 1.0, 1.0, 1.0, 1.2, 1.6])
# ===============================================================
# --- FUNCIONES DEL MODELO DE BATERÍA ---
# ===============================================================

def get_ocv(soc, temp, mode='discharge'):
    """
    Calcula el OCV basado en SOC, temperatura y modo (carga/descarga).
    Utiliza el modelo y los parámetros de la Tabla 7 de Baccouche et al. (2017).
    """
    soc_percent = soc * 100

    if mode == 'charge':
        params_table = ocv_params_charge
    else:
        params_table = ocv_params_discharge
        
    # Interpolar cada parámetro del modelo OCV basado en la temperatura
    p0, p1, p2, a1, a2 = [np.interp(temp, ocv_temperatures, params_table[:, i]) for i in range(5)]
    
    ocv = p0 * np.exp(a1 * soc_percent) + p1 * np.exp(a2 * soc_percent) + p2 * soc_percent**2
    return ocv


def get_rbatt(soc, temp):
    """
    Calcula la resistencia interna efectiva de un PAQUETE de baterías.
    
    Se aplica un factor de escala para modelar un paquete de mayor capacidad
    (ej. 3 celdas de 20Ah en paralelo), alineando el modelo de batería con
    la escala de la aplicación de vehículo de pasajeros.
    """
    # Factor de escala para simular un paquete de 60Ah (3 x 20Ah en paralelo)
    SCALE_FACTOR = 3
    
    # Interpolar R0 y R1 basados en la temperatura para una sola celda
    r0_cell_temp = np.interp(temp, res_temperatures, r0_table)
    r1_cell_temp = np.interp(temp, res_temperatures, r1_table)
    r_base_cell = r0_cell_temp + r1_cell_temp

    soc_multiplier = 1
    # La resistencia total de la celda es la suma
    r_total_cell = r_base_cell * soc_multiplier    
    # La resistencia del paquete es la de la celda dividida por el número de strings en paralelo
    r_total_pack = r_total_cell / SCALE_FACTOR
    
    return r_total_pack


def get_cnom(temp):
    """
    Calcula la capacidad nominal de la batería (C_nom) en Ah.
    Interpola la capacidad en función de la temperatura, basado en la Tabla 2
    de Baccouche et al. (2017).
    
    Args:
        temp (float): Temperatura actual de la batería [°C].

    Returns:
        float: Capacidad nominal en Amperios-hora [Ah].
    """
    return np.interp(temp, res_temperatures, cnom_table)

try:
    processed_data_path = r'C:\Users\super\Desktop\Vasudeva\Manifold\engineering\predictive-control\stochastic-mpc\entropy_model_data.npz'
    entropy_data = np.load(processed_data_path)
    soc_points_dvdt = entropy_data['soc_points']
    dvdt_points = entropy_data['dvdt_points']
except FileNotFoundError:
    print("Error: El archivo 'entropy_model_data.npz' no fue encontrado.")
    print("Por favor, ejecuta primero el script 'process_entropy_data.py'.")
    # Define arrays vacíos para evitar que el programa se caiga al importar
    soc_points_dvdt = np.array([0, 1])
    dvdt_points = np.array([0, 0])
    
def get_dvdt(soc):
    """
    Calcula el coeficiente de calor entrópico (dV/dT) para un SOC dado.
    Utiliza interpolación lineal sobre los datos pre-procesados.
    """
    return np.interp(soc, soc_points_dvdt, dvdt_points)