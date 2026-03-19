import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path
import jax.numpy as jnp
from dataclasses import dataclass

PLOT_CONFIG = {
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
    "lines.linewidth": 1.4,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linestyle": "--",
    "savefig.dpi": 300
}

plt.rcParams.update(PLOT_CONFIG)

@dataclass
class FigConfig:
    vertical = {
        "figsize": (4.0, 10.0),
        "nrows": 5,
        "ncols":1,
        "sharex":True,
        "layout":"constrained"
    }
    horizontal = {
        "figsize":(13.0, 2.5),
        "nrows": 1,
        "ncols":5,
        "layout":"constrained"
    }

def label_plot(ax, title: str, config: str):
    if config == 'vertical':
        ax.set_ylabel(title)
    elif config == 'horizontal':
        ax.set_title(title)
        ax.set_xlabel('Tiempo (s)')
    
def plot_results(df_controller: pd.DataFrame, name: str, config: str, dt=1.0): # Added dt argument

    assert config in ('vertical', 'horizontal')

    time = df_controller['time']
    # Extract data
    Q_cool = df_controller['Q_cool']
    # Q_gen = df_controller['Q_gen'] # Un-comment if you want to plot this too
    T_batt = df_controller['T_batt']
    T_clnt = df_controller['T_clnt']
    w_pump = df_controller['w_pump']
    w_comp = df_controller['w_comp']
    P_cooling = df_controller['P_cooling']

    if config == 'vertical':
        fig, axs = plt.subplots(**FigConfig.vertical)
    elif config == 'horizontal':
        fig, axs = plt.subplots(**FigConfig.horizontal)
    
    # --- 1. Heat Transfer ---
    axs[0].plot(time, Q_cool, 'b', lw=1.5)
    label_plot(axs[0], r'Calor Removido' + '\n' + r'($\dot{Q}_{cool}$) [W]', config) 
    axs[0].set_xlim(0, len(time))
    axs[0].set_ylim(0, 2000)
    
    # --- 2. Pump Speed ---
    axs[1].plot(time, w_pump, 'r')
    label_plot(axs[1], r'Vel. Bomba' + '\n' + r'($\omega_{pump}$) [RPM]', config)
    axs[1].set_xlim(0, len(time))
    axs[1].set_ylim(0, 10000)

    # --- 3. Compressor Speed ---
    axs[2].plot(time, w_comp, 'k')
    label_plot(axs[2], r'Vel. Compresor' + '\n' + r'($\omega_{comp}$) [RPM]', config)
    axs[2].set_xlim(0, len(time))
    axs[2].set_ylim(0, 10000)
    
    # --- 4. Temperatures ---
    axs[3].plot(time, T_batt, 'r', label='$T_{batt}$')
    axs[3].plot(time, T_clnt, 'b--', label='$T_{clnt}$')
    label_plot(axs[3], r'Temperatura' + '\n' + r'($T$) [$^\circ$C]', config)
    axs[3].legend(loc='upper left', frameon=True) 
    axs[3].set_xlim(0, len(time))
    axs[3].set_ylim(28, 35)

    # --- 5. Energy Consumption ---
    joules_to_kJ = 1/1000
    energy_cooling_kJ = np.cumsum(P_cooling) * dt * joules_to_kJ
    
    axs[4].plot(time, energy_cooling_kJ, 'g')
    if config == 'vertical':
        axs[4].set_xlabel('Tiempo (s)')
    label_plot(axs[4], 'Energia de Enf.' + '\n' + r'($P_{cool}$) [kJ]', config)
    axs[4].grid(True, which='both')
    axs[4].set_xlim(0, len(time))
    axs[4].set_ylim(0, 400)
    
    save_dir = Path('results')    
    save_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_dir / f"{name}_controller.png", bbox_inches='tight', dpi=300)
    plt.show()

def show_results(
        states_hist: np.ndarray | jnp.ndarray = None, 
        ctrl_hist: np.ndarray | jnp.ndarray = None, 
        diag_hist: np.ndarray | jnp.ndarray = None,
        controller_name: str = 'any',
        df: pd.DataFrame = None,
        config: str = 'vertical'
    ):
    if df is not None:
        N = len(df)
        df = df.copy()
    else: 
        N = len(states_hist)
        df = pd.DataFrame({
            'time': np.arange(N),
            # Estados
            'T_batt': states_hist[:, 0],
            'T_clnt': states_hist[:, 1],
            # Controles
            'w_comp': ctrl_hist[:, 0],
            'w_pump': ctrl_hist[:, 1],
            # Diagnósticos 
            'P_cooling': diag_hist[:, 0],
            'Q_gen':     diag_hist[:, 4],
            'Q_cool':    diag_hist[:, 5]
        })
    print(f"Total Energy: {(df['P_cooling'].sum()/1000):.4f} kJ")
    print(f"Final T_batt: \n{df[['time','T_batt']].tail(3)}")
    
    plot_results(df, controller_name, config)


def plot_learning_history(history):
    episodes = np.arange(len(history['ep_rewards']))
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 5), sharex=True)

    ax1.plot(episodes, history['ep_energy_kj'], color='dodgerblue')
    ax1.set_ylabel('Energia'+ '\n' + r'Consumida [kJ]')

    ax2.plot(episodes, history['ep_avg_temp'], color='red')
    ax2.set_ylabel(r'Promedio $T_{batt}$ [°C]')
    plt.tight_layout()
    plt.show()


    fig, ax = plt.subplots(figsize=(7, 5))

    ax.plot(episodes[5:], history['ep_rewards'][5:], color='seagreen')
    ax.set_ylabel('Recompensa'+ '\n' + r'Cumulativa ($R$)')
    ax.set_xlabel('Episodio')
    ax.ticklabel_format(axis='y')
    plt.tight_layout()
    plt.show()


def plot_sensitivity(
    df_base: pd.DataFrame,
    df_minus: pd.DataFrame,
    df_plus: pd.DataFrame,
    param_name: str,       # ej. r'$R_{int}$'
    name: str,
    config: str,
    dt: float = 1.0,
):
    assert config in ('vertical', 'horizontal')

    COLORS  = {'minus': '#E8593C', 'base': '#444441', 'plus': '#3B8BD4'}
    STYLES  = {'minus': '--',      'base': '-',        'plus': ':'}
    LABELS  = {'minus': '-20%',    'base': 'Base',     'plus': '+20%'}
    runs    = {'minus': df_minus,  'base': df_base,    'plus': df_plus}

    time = df_base['time'].values

    if config == 'vertical':
        fig, axs = plt.subplots(**FigConfig.vertical)
    else:
        fig, axs = plt.subplots(**FigConfig.horizontal)

    def plot_three(ax, col, color_key=None):
        for key, df in runs.items():
            c = COLORS[color_key or key] if color_key is None else COLORS[key]
            ax.plot(time, df[col].values,
                    color=COLORS[key], lw=1.4,
                    linestyle=STYLES[key], label=LABELS[key])

    # 1. Q_cool
    plot_three(axs[0], 'Q_cool')
    label_plot(axs[0], r'Calor Removido' + '\n' + r'($\dot{Q}_{cool}$) [W]', config)
    axs[0].set_xlim(0, time[-1]); axs[0].set_ylim(0, 2000)

    # 2. w_pump
    plot_three(axs[1], 'w_pump')
    label_plot(axs[1], r'Vel. Bomba' + '\n' + r'($\omega_{pump}$) [RPM]', config)
    axs[1].set_xlim(0, time[-1]); axs[1].set_ylim(0, 10000)

    # 3. w_comp
    plot_three(axs[2], 'w_comp')
    label_plot(axs[2], r'Vel. Compresor' + '\n' + r'($\omega_{comp}$) [RPM]', config)
    axs[2].set_xlim(0, time[-1]); axs[2].set_ylim(0, 10000)

    # 4. T_batt and T_clnt
    for key, df in runs.items():
        axs[3].plot(time, df['T_batt'].values,
                    color=COLORS[key], lw=1.4, linestyle=STYLES[key],
                    label=f'$T_{{batt}}$ {LABELS[key]}')
        axs[3].plot(time, df['T_clnt'].values,
                    color=COLORS[key], lw=0.8, linestyle=STYLES[key], alpha=0.5,
                    label=f'$T_{{clnt}}$ {LABELS[key]}')
    label_plot(axs[3], r'Temperatura' + '\n' + r'($T$) [$^\circ$C]', config)
    axs[3].legend(loc='upper left', frameon=True, fontsize=7, ncol=2)
    axs[3].set_xlim(0, time[-1]); axs[3].set_ylim(28, 35)

    # 5. Cumulative energy
    for key, df in runs.items():
        energy = np.cumsum(df['P_cooling'].values) * dt / 1000
        axs[4].plot(time, energy,
                    color=COLORS[key], lw=1.4,
                    linestyle=STYLES[key], label=LABELS[key])
    if config == 'vertical':
        axs[4].set_xlabel('Tiempo (s)')
    label_plot(axs[4], 'Energia de Enf.' + '\n' + r'($P_{cool}$) [kJ]', config)
    axs[4].set_xlim(0, time[-1]); axs[4].set_ylim(0, 400)

    # Global legend
    axs[0].legend(title=param_name, frameon=True, fontsize=8)


    save_dir = Path('results')
    save_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_dir / f"{name}_sensitivity.png", bbox_inches='tight', dpi=300)
    plt.show()