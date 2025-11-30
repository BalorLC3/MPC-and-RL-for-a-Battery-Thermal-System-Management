import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# --- 1. IMPORTACIONES (sin cambios) ---
from efficiency import (get_volumetric_eff, get_isentropic_eff, get_motor_eff, 
                        get_pump_pressure_drop, PUMP_MAX_SPEED_RPM, COMP_MAX_SPEED_RPM)
from battery_models import get_ocv, get_rbatt, get_cnom, get_dvdt

# --- 2. CLASE DE PARÁMETROS (sin cambios) ---
class SystemParameters:
    def __init__(self):
        self.rho_rfg = 27.8
        self.rho_clnt = 1069.5
        self.C_rfg = 1117.0
        self.C_clnt = 3330.0
        self.V_comp = 33e-6
        self.V_pump = 33e-6
        self.h_eva = 1000.0
        self.A_eva = 0.3
        self.h_batt = 300.0
        self.A_batt = 1.0
        self.PR = 5.0
        self.h_cout_kJ = 284.3
        self.h_evaout_kJ = 250.9
        self.m_batt = 40.0
        self.C_batt = 1350.0
        self.m_clnt_total = 2.0 * self.rho_clnt / 1000

        # Flujo minimo de refrigerante para evitar m_clnt_dot == 0
        self.m_clnt_dot_min = 1e-4

# ===============================================================
# === CLASE PRINCIPAL DEL ENTORNO (CON MODIFICACIONES) ===
# ===============================================================
class BatteryThermalSystem:
    def __init__(self, initial_state, params):
        self.params = params
        self.state = np.array([
            initial_state['T_batt'],
            initial_state['T_clnt'],
            initial_state['soc']
        ])
        # Almacenará el último diccionario de diagnóstico
        self.diagnostics = {}

    def _system_dynamics(self, t, y, u, d):
        T_batt, T_clnt, soc = y[0], y[1], y[2]
        w_comp, w_pump = u[0], u[1]
        P_driv, T_amb = d[0], d[1]

        # --- Modelo del Sistema de Enfriamiento ---
        eta_vol_pump = get_volumetric_eff(w_pump, PUMP_MAX_SPEED_RPM, 0.98)
        m_clnt_dot = self.params.V_pump * (w_pump / 60) * eta_vol_pump * self.params.rho_clnt
        # aplicar flujo mínimo para evitar m_clnt_dot == 0 (suaviza el modelo)
        m_clnt_dot = max(m_clnt_dot, self.params.m_clnt_dot_min)

        delta_p_pump = get_pump_pressure_drop(m_clnt_dot)
        eta_p_motor = get_motor_eff(w_pump)
        P_pump_mech = (m_clnt_dot * delta_p_pump) / self.params.rho_clnt if self.params.rho_clnt > 0 else 0
        P_pump_elec = P_pump_mech / eta_p_motor if eta_p_motor > 0 else 0

        eta_vol_comp = get_volumetric_eff(w_comp, COMP_MAX_SPEED_RPM, 0.95)
        m_rfg_dot = self.params.V_comp * (w_comp / 60) * eta_vol_comp * self.params.rho_rfg 
        eta_isen = get_isentropic_eff(w_comp)
        eta_c_motor = get_motor_eff(w_comp)
        h_delta_J = (self.params.h_cout_kJ - self.params.h_evaout_kJ) * 1000
        P_comp_mech = (m_rfg_dot * h_delta_J) / eta_isen if eta_isen > 0 else 0
        P_comp_elec = P_comp_mech / eta_c_motor if eta_c_motor > 0 else 0
        P_cooling = P_pump_elec + P_comp_elec

        # --- Modelo Eléctrico y Térmico de la Batería ---
        P_aux = 50
        P_batt_total = P_driv + P_cooling + P_aux  # W

        # Determinar modo OCV según sentido de la corriente real
        ocv_mode = 'charge' if P_batt_total < 0 else 'discharge'
        V_oc = get_ocv(soc, T_batt, mode=ocv_mode)
        R_batt = get_rbatt(soc, T_batt)

        # === LIMITES FÍSICOS DE CORRIENTE ===
        I_max_discharge = 2.5 * get_cnom(T_batt)     
        I_max_charge    = 1.0 * get_cnom(T_batt)    

        # === Corriente ideal según ecuación cuadrática ===
        discriminant = V_oc**2 - 4 * R_batt * P_batt_total
        if R_batt > 0 and discriminant >= 0:
            I_batt = (V_oc - np.sqrt(discriminant)) / (2 * R_batt)
        else:
            I_batt = 0.0

        # === Bloqueo de regeneración cuando SOC está cerca de 100% ===
        if soc >= 0.995 and I_batt < 0:
            I_batt = 0.0

        # === Bloqueo de descarga cuando SOC está en 0% ===
        if soc <= 0.005 and I_batt > 0:
            I_batt = 0.0

        # === Aplicar límites de corriente (saturación física) ===
        I_batt = np.clip(I_batt, -I_max_charge, I_max_discharge)

        # === Recalcular potencia real *después* de límites ===
        P_batt_total = V_oc * I_batt

        # === Térmica: generación de calor ===
        dVdT_batt = get_dvdt(soc)
        T_batt_kelvin = T_batt + 273.15
        Q_gen = I_batt**2 * R_batt - I_batt * T_batt_kelvin * dVdT_batt


        # --- Modelo de Transferencia de Calor ---
        T_clnt_chilled = self._model_evaporator(T_clnt, m_clnt_dot, m_rfg_dot)
        T_clnt_hot, Q_cool = self._model_battery_cooling(T_batt, T_clnt_chilled, m_clnt_dot)

        # --- Derivadas ---
        dT_batt_dt = (Q_gen - Q_cool) / (self.params.m_batt * self.params.C_batt)
        heat_gain_clnt = Q_cool
        heat_loss_clnt = m_clnt_dot * self.params.C_clnt * (T_clnt - T_clnt_chilled)
        dT_clnt_dt = (heat_gain_clnt - heat_loss_clnt) / (self.params.m_clnt_total * self.params.C_clnt)
        C_nom_Ah = get_cnom(T_batt)
        Qn_As = C_nom_Ah * 3600
        dSOC_dt = -I_batt / Qn_As if Qn_As > 0 else 0
        # =================
        # TELEMETRIA: POTENCIA IN & OUT
        # =================
        P_out = P_batt_total if P_batt_total > 0 else 0
        P_in = -P_batt_total if P_batt_total < 0 else 0

        self.diagnostics = {
            'P_cooling': P_cooling, 'P_batt_total': P_batt_total,
            'P_max': (V_oc**2) / (4 * R_batt) if R_batt > 0 else 0,
            'V_oc': V_oc, 'R_batt': R_batt,
            'discriminant': discriminant, 'I_batt': I_batt,
            'Q_gen': Q_gen, 'Q_cool': Q_cool,
            'P_in': P_in,    
            'P_out': P_out   
        }
        
        return [dT_batt_dt, dT_clnt_dt, dSOC_dt]

    def step(self, controls, disturbances, dt):
        t_span = [0, dt]
        sol = solve_ivp(
            fun=self._system_dynamics, t_span=t_span, y0=self.state,
            args=(controls, disturbances), method='RK45'
        )
        self.state = sol.y[:, -1]
        # Devolvemos el estado Y el diccionario de diagnóstico
        return self.state, self.diagnostics

    # --- Métodos internos ---
    def _model_evaporator(self, T_clnt_in, m_clnt_dot, m_rfg_dot):
        T_rfg_in = 1.2
        if m_clnt_dot <= self.params.m_clnt_dot_min or m_rfg_dot <= 0:
            return T_clnt_in

        C_clnt_dot = m_clnt_dot * self.params.C_clnt
        C_rfg_dot = m_rfg_dot * self.params.C_rfg
        C_min = min(C_clnt_dot, C_rfg_dot)
        C_max = max(C_clnt_dot, C_rfg_dot)
        Cr = C_min / C_max
        UA = self.params.h_eva * self.params.A_eva
        NTU = UA / C_min
        effectiveness = (1 - np.exp(-NTU * (1 + Cr))) / (1 + Cr)
        Q_max = C_min * (T_clnt_in - T_rfg_in)
        Q_actual = effectiveness * Q_max
        T_clnt_out = T_clnt_in - (Q_actual / C_clnt_dot)
        return T_clnt_out


    def _model_battery_cooling(self, T_batt, T_clnt_in, m_clnt_dot):
        if m_clnt_dot <= 0: return T_clnt_in, 0
        exponent = -(self.params.h_batt * self.params.A_batt) / (m_clnt_dot * self.params.C_clnt)
        T_clnt_out = T_batt - (T_batt - T_clnt_in) * np.exp(exponent)
        Q_cool = m_clnt_dot * self.params.C_clnt * (T_clnt_out - T_clnt_in)
        return T_clnt_out, Q_cool

import numpy as np


try:
    driving_data = np.load('driving_energy.npy', mmap_mode='r')
except FileNotFoundError:
    print("Error: NO se encontro el archivo del perfil de conduccion")

if __name__ == "__main__":
    params = SystemParameters()
    initial_state = {'T_batt': 35.0, 'T_clnt': 30.0, 'soc': 0.8}
    env = BatteryThermalSystem(initial_state, params)    
    controls = [300.0, 3000.0]

    T_ambient = 25.0
    dt = 1.0
    sim_time = len(driving_data)
    time_steps = np.arange(0, sim_time, dt)
    history = []
    diagnostics_history = []

    for i, t in enumerate(time_steps):
        # potencia de manejo del perfil para tiempoa actual
        current_p_driv = driving_data[i]
        # Crear vector de perturbaciones para este step
        current_disturbances = [current_p_driv, T_ambient]
        history.append(env.state)
        if i % 100 == 0: 
            # Naivy control
            controls = [x - np.pi * i // 10 for x in controls]
            if any(control <= 0 for control in controls):
                controls = [0, 0]
        _, diagnostics = env.step(controls, current_disturbances, dt)
        diagnostics_history.append(diagnostics)
    
    history = np.array(history)

    diagnostics_log = {k: np.array([d[k] for d in diagnostics_history]) for k in diagnostics_history[0]}
    

    fig, axs = plt.subplots(4, 1, figsize=(12, 12), sharex=True)
    
    axs[0].plot(time_steps, driving_data / 1000, 'k-', alpha=0.8)
    axs[0].set_title('Perfil de Potencia de Manejo (Perturbación)')
    axs[0].set_ylabel('P_driv (kW)')
    axs[0].grid(True)
    
    axs[1].plot(time_steps, history[:, 0], label='T Batería')
    axs[1].plot(time_steps, history[:, 1], label='T Refrigerante')
    axs[1].set_ylabel('Temperatura (°C)')
    axs[1].legend(); axs[1].grid(True)

    axs[2].plot(time_steps, history[:, 2] * 100)
    axs[2].set_ylabel('SOC (%)'); axs[2].grid(True)

    axs[3].plot(time_steps, diagnostics_log['I_batt'])
    axs[3].set_ylabel('Corriente (A)'); axs[3].grid(True)
    axs[3].set_xlabel('Tiempo (s)')

    plt.suptitle('Simulacion de Lazo Abierto (Ciclo UDDS)', fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()