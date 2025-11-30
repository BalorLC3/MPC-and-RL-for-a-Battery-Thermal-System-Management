import numpy as np
import matplotlib.pyplot as plt

# ===============================================================
# GENERIC COOLING SYSTEM COMPONENT MODELS
#
# Este archivo proporciona modelos paramétricos para
# las eficiencias de la bomba y el compresor, y para la caída de presión
# en la bomba. Estas funciones reemplazan la necesidad de extraer datos
# ===============================================================

# --- Parámetros y Suposiciones de los Componentes ---
# Estos valores definen el comportamiento de nuestros componentes genéricos.
# Se eligen para ser representativos de un sistema de automoción.

# Parámetros del Compresor
COMP_MAX_SPEED_RPM = 10000.0
COMP_NOMINAL_SPEED_RPM = 6000.0  # Velocidad de máxima eficiencia
COMP_MAX_VOL_EFF = 0.95         # Eficiencia volumétrica a baja velocidad
COMP_MAX_ISEN_EFF = 0.80        # Eficiencia isentrópica máxima

# Parámetros de la Bomba
PUMP_MAX_SPEED_RPM = 8000.0
PUMP_MAX_VOL_EFF = 0.98
# Coeficiente para la caída de presión. Se ajusta para que a un caudal máximo
# razonable (ej. ~3 kg/s), la presión sea de ~30 kPa, similar a la Figura 3(d).
PUMP_PRESSURE_COEFF = 3300.0  # Pa / (kg/s)^2

# Parámetros del Motor Eléctrico (asumido igual para ambos)
MOTOR_MAX_EFF = 0.92
MOTOR_NOMINAL_SPEED_RPM = 5000.0


def get_volumetric_eff(speed_rpm, max_speed_rpm, max_eff):
    """
    Modelo genérico para la eficiencia volumétrica.
    Justificación: La eficiencia volumétrica típicamente es máxima a baja velocidad y
    decae linealmente a medida que aumentan las pérdidas por fricción y fugas a
    altas velocidades.
    
    Args:
        speed_rpm (float): Velocidad actual del componente en RPM.
        max_speed_rpm (float): Velocidad máxima del componente.
        max_eff (float): Eficiencia máxima a velocidad cero.

    Returns:
        float: Eficiencia volumétrica (0 a 1).
    """
    if speed_rpm <= 0:
        return 0.0
    
    # Modelo lineal simple: eff = max_eff - slope * (speed / max_speed)
    slope = 0.4 # Factor de decaimiento
    efficiency = max_eff - slope * (speed_rpm / max_speed_rpm)
    
    # Asegurarse de que la eficiencia no sea negativa
    return np.clip(efficiency, 0.0, max_eff)


def get_isentropic_eff(speed_rpm):
    """
    Modelo genérico para la eficiencia isentrópica del compresor.
    Justificación: La eficiencia isentrópica no es monótona. Es baja a bajas
    velocidades, alcanza un pico en un punto de diseño nominal y vuelve a caer a
    velocidades muy altas. Una parábola invertida (función cuadrática) es un
    excelente modelo para este comportamiento.
    """
    if speed_rpm <= 0:
        return 0.0

    # Modelo cuadrático: eff = max_eff - k * (speed - nominal_speed)^2
    # El término de velocidad se normaliza para un mejor comportamiento
    norm_speed_diff = (speed_rpm - COMP_NOMINAL_SPEED_RPM) / COMP_MAX_SPEED_RPM
    k = 0.5 # Factor que controla la anchura de la parábola
    
    efficiency = COMP_MAX_ISEN_EFF - k * (norm_speed_diff**2)
    
    return np.clip(efficiency, 0.0, COMP_MAX_ISEN_EFF)


def get_motor_eff(speed_rpm):
    """
    Modelo genérico para la eficiencia del motor eléctrico.
    Justificación: Similar a la eficiencia isentrópica, los motores eléctricos
    tienen un punto de operación de máxima eficiencia.
    """
    if speed_rpm <= 0:
        return 0.0

    norm_speed_diff = (speed_rpm - MOTOR_NOMINAL_SPEED_RPM) / COMP_MAX_SPEED_RPM
    k = 0.4
    
    efficiency = MOTOR_MAX_EFF - k * (norm_speed_diff**2)
    
    return np.clip(efficiency, 0.0, MOTOR_MAX_EFF)


def get_pump_pressure_drop(m_clnt_dot):
    """
    Modelo genérico para la caída de presión en la bomba.
    Justificación: Según los principios de la dinámica de fluidos, la caída de
    presión en un sistema de tuberías es aproximadamente proporcional al cuadrado
    del caudal másico (ΔP ∝ ṁ^2).
    
    Args:
        m_clnt_dot (float): Caudal másico de refrigerante [kg/s].

    Returns:
        float: Caída de presión en Pascales [Pa].
    """
    return PUMP_PRESSURE_COEFF * m_clnt_dot**2

