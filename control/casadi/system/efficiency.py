import casadi as ca

def _clip(val, min_val, max_val):
    """Equivalent to jnp.clip using CasADi primitives"""
    return ca.fmin(ca.fmax(val, min_val), max_val)


def get_volumetric_eff(speed_rpm, params):
    s = ca.fmax(speed_rpm, 0.0)
    slope = 0.4
    eff = params.comp_max_vol_eff - slope * (s / params.comp_max_speed_rpm)
    return _clip(eff, 0.0, params.comp_max_vol_eff)


def get_isentropic_eff(speed_rpm, params):
    s = ca.fmax(speed_rpm, 0.0)
    norm_speed_diff = (s - params.comp_nominal_speed_rpm) / params.comp_max_speed_rpm
    k = 0.5
    eff = params.comp_max_isen_eff - k * (norm_speed_diff ** 2)
    return _clip(eff, 0.0, params.comp_max_isen_eff)


def get_motor_eff(speed_rpm, params):
    s = ca.fmax(speed_rpm, 0.0)
    norm_speed_diff = (s - params.motor_nominal_speed_rpm) / params.comp_max_speed_rpm
    k = 0.4
    eff = params.motor_max_eff - k * (norm_speed_diff ** 2)
    return _clip(eff, 0.0, params.motor_max_eff)


def get_pump_pressure_drop(m_clnt_dot, params):
    m = ca.fmax(m_clnt_dot, 0.0)
    return params.pump_pressure_coeff * (m ** 2)