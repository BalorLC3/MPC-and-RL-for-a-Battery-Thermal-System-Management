import jax.numpy as jnp

# ===============================================================
# GENERIC COOLING SYSTEM COMPONENT MODELS
# ===============================================================

# Compressor
def get_volumetric_eff(speed_rpm, params):
    s = jnp.maximum(speed_rpm, 0.0)
    slope = 0.4
    eff = params.comp_max_vol_eff - slope * (s / params.comp_max_speed_rpm)
    return jnp.clip(eff, 0.0, params.comp_max_vol_eff)


def get_isentropic_eff(speed_rpm, params):
    s = jnp.maximum(speed_rpm, 0.0)
    norm_speed_diff = (s - params.comp_nominal_speed_rpm) / params.comp_max_speed_rpm
    k = 0.5
    eff = params.comp_max_isen_eff - k * (norm_speed_diff ** 2)
    return jnp.clip(eff, 0.0, params.comp_max_isen_eff)


def get_motor_eff(speed_rpm, params):
    s = jnp.maximum(speed_rpm, 0.0)
    norm_speed_diff = (s - params.motor_nominal_speed_rpm) / params.comp_max_speed_rpm
    k = 0.4
    eff = params.motor_max_eff - k * (norm_speed_diff ** 2)
    return jnp.clip(eff, 0.0, params.motor_max_eff)


def get_pump_pressure_drop(m_clnt_dot, params):
    m = jnp.maximum(m_clnt_dot, 0.0)
    return params.pump_pressure_coeff * (m ** 2)

