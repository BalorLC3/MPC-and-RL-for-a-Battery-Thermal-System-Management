import casadi as ca
import numpy as np

# ===============================================================
# GENERIC COOLING SYSTEM COMPONENT MODELS (CASADI VERSION)
# ===============================================================

# Parámetros del Compresor
COMP_MAX_SPEED_RPM = 10000.0
COMP_NOMINAL_SPEED_RPM = 6000.0  # Velocidad de máxima eficiencia
COMP_MAX_VOL_EFF = 0.95         # Eficiencia volumétrica a baja velocidad
COMP_MAX_ISEN_EFF = 0.80        # Eficiencia isentrópica máxima

# Parámetros de la Bomba
PUMP_MAX_SPEED_RPM = 8000.0
PUMP_MAX_VOL_EFF = 0.98
PUMP_PRESSURE_COEFF = 3300.0  # Pa / (kg/s)^2

# Parámetros del Motor Eléctrico
MOTOR_MAX_EFF = 0.92
MOTOR_NOMINAL_SPEED_RPM = 5000.0

# --- Helper interno para replicar jnp.clip ---
def _clip(val, min_val, max_val):
    """Equivalente a jnp.clip(val, min, max) usando primitivas CasADi"""
    return ca.fmin(ca.fmax(val, min_val), max_val)

def get_volumetric_eff(speed_rpm, max_speed_rpm=COMP_MAX_SPEED_RPM, max_eff=COMP_MAX_VOL_EFF):
    # ca.fmax es el equivalente a jnp.maximum
    s = ca.fmax(speed_rpm, 0.0)
    
    slope = 0.4
    # Evitamos división por cero si max_speed_rpm fuera simbólico (aunque aquí es cte)
    eff = max_eff - slope * (s / (max_speed_rpm + 1e-9))
    
    return _clip(eff, 0.0, max_eff)


def get_isentropic_eff(speed_rpm):
    s = ca.fmax(speed_rpm, 0.0)
    
    norm_speed_diff = (s - COMP_NOMINAL_SPEED_RPM) / COMP_MAX_SPEED_RPM
    k = 0.5
    eff = COMP_MAX_ISEN_EFF - k * (norm_speed_diff ** 2)
    
    return _clip(eff, 0.0, COMP_MAX_ISEN_EFF)


def get_motor_eff(speed_rpm):
    s = ca.fmax(speed_rpm, 0.0)
    
    norm_speed_diff = (s - MOTOR_NOMINAL_SPEED_RPM) / COMP_MAX_SPEED_RPM
    k = 0.4
    eff = MOTOR_MAX_EFF - k * (norm_speed_diff ** 2)
    
    return _clip(eff, 0.0, MOTOR_MAX_EFF)


def get_pump_pressure_drop(m_clnt_dot):
    m = ca.fmax(m_clnt_dot, 0.0)
    return PUMP_PRESSURE_COEFF * (m ** 2)