import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path
import jax.numpy as jnp
from dataclasses import dataclass

PLOT_CONFIG = {
    "text.usetex": False,
    "font.family": "sans-serif",
    "font.sans-serif": ["Microsoft YaHei"],
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
        "ncols": 1,
        "sharex": True,
        "layout": "constrained"
    }
    horizontal = {
        "figsize": (13.0, 2.5),
        "nrows": 1,
        "ncols": 5,
        "layout": "constrained"
    }


# ---------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------

def label_plot(ax, title: str, config: str):
    if config == 'vertical':
        ax.set_ylabel(title)
    elif config == 'horizontal':
        ax.set_title(title)
        ax.set_xlabel('Tiempo (s)')


def _hist_to_df(hist: dict) -> pd.DataFrame:
    """Converts a JAX simulation history dict to a DataFrame."""
    states_hist = np.asarray(hist['state'])
    ctrl_hist   = np.asarray(hist['controls'])
    diag_hist   = np.asarray(hist['diagnostics'])
    return pd.DataFrame({
        'time':      np.arange(len(states_hist)),
        'T_batt':    states_hist[:, 0],
        'T_clnt':    states_hist[:, 1],
        'w_comp':    ctrl_hist[:, 0],
        'w_pump':    ctrl_hist[:, 1],
        'P_cooling': diag_hist[:, 0],
        'Q_gen':     diag_hist[:, 4],
        'Q_cool':    diag_hist[:, 5],
    })


def _save(name: str, suffix: str):
    save_dir = Path('results')
    save_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_dir / f"{name}_{suffix}.png", bbox_inches='tight', dpi=300)


# ---------------------------------------------------------------
# Core plot functions
# ---------------------------------------------------------------

def plot_results(df: pd.DataFrame, name: str, config: str, dt: float = 1.0):
    assert config in ('vertical', 'horizontal')

    time      = df['time']
    Q_cool    = df['Q_cool']
    T_batt    = df['T_batt']
    T_clnt    = df['T_clnt']
    w_pump    = df['w_pump']
    w_comp    = df['w_comp']
    P_cooling = df['P_cooling']

    fig, axs = plt.subplots(**FigConfig.vertical if config == 'vertical' else FigConfig.horizontal)

    axs[0].plot(time, Q_cool, 'b', lw=1.5)
    label_plot(axs[0], r'Calor Removido' + '\n' + r'($\dot{Q}_{cool}$) [W]', config)
    axs[0].set_xlim(0, len(time)); axs[0].set_ylim(0, 2000)

    axs[1].plot(time, w_pump, 'r')
    label_plot(axs[1], r'Vel. Bomba' + '\n' + r'($\omega_{pump}$) [RPM]', config)
    axs[1].set_xlim(0, len(time)); axs[1].set_ylim(0, 10000)

    axs[2].plot(time, w_comp, 'k')
    label_plot(axs[2], r'Vel. Compresor' + '\n' + r'($\omega_{comp}$) [RPM]', config)
    axs[2].set_xlim(0, len(time)); axs[2].set_ylim(0, 10000)

    axs[3].plot(time, T_batt, 'r',  label='$T_{batt}$')
    axs[3].plot(time, T_clnt, 'b--', label='$T_{clnt}$')
    label_plot(axs[3], r'Temperatura' + '\n' + r'($T$) [$^\circ$C]', config)
    axs[3].legend(loc='upper left', frameon=True)
    axs[3].set_xlim(0, len(time)); axs[3].set_ylim(28, 35)

    energy_kJ = np.cumsum(P_cooling) * dt / 1000
    axs[4].plot(time, energy_kJ, 'g')
    if config == 'vertical':
        axs[4].set_xlabel('Tiempo (s)')
    label_plot(axs[4], 'Energia de Enf.' + '\n' + r'($P_{cool}$) [kJ]', config)
    axs[4].set_xlim(0, len(time)); axs[4].set_ylim(0, 400)

    _save(name, 'controller')
    plt.show()


def plot_sensitivity(
    df_minus: pd.DataFrame,
    df_base:  pd.DataFrame,
    df_plus:  pd.DataFrame,
    param_name: str,
    name: str,
    config: str,
    dt: float = 1.0,
):
    assert config in ('vertical', 'horizontal')

    time = df_base['time'].values if 'time' in df_base else np.arange(len(df_base))

    fig, axs = plt.subplots(**FigConfig.vertical if config == 'vertical' else FigConfig.horizontal)

    ALPHA = 0.25

    def shade(ax, col, color):
        lo  = df_minus[col].values
        mid = df_base[col].values
        hi  = df_plus[col].values
        ax.fill_between(time, lo, hi, color=color, alpha=ALPHA, label='±20%')
        ax.plot(time, lo,  color=color, lw=0.8, linestyle='--', alpha=0.6)
        ax.plot(time, hi,  color=color, lw=0.8, linestyle='--', alpha=0.6)
        ax.plot(time, mid, color='black', lw=1.4, label='Base')

    shade(axs[0], 'Q_cool', 'steelblue')
    label_plot(axs[0], r'Calor Removido' + '\n' + r'($\dot{Q}_{cool}$) [W]', config)
    axs[0].set_xlim(0, time[-1]); axs[0].set_ylim(0, 2000)
    axs[0].legend(frameon=True)

    shade(axs[1], 'w_pump', 'tomato')
    label_plot(axs[1], r'Vel. Bomba' + '\n' + r'($\omega_{pump}$) [RPM]', config)
    axs[1].set_xlim(0, time[-1]); axs[1].set_ylim(0, 10000)

    shade(axs[2], 'w_comp', 'slategray')
    label_plot(axs[2], r'Vel. Compresor' + '\n' + r'($\omega_{comp}$) [RPM]', config)
    axs[2].set_xlim(0, time[-1]); axs[2].set_ylim(0, 10000)

    for col, color, lbl in [('T_batt', 'tomato', '$T_{batt}$'), ('T_clnt', 'steelblue', '$T_{clnt}$')]:
        lo  = df_minus[col].values
        mid = df_base[col].values
        hi  = df_plus[col].values
        axs[3].fill_between(time, lo, hi, color=color, alpha=ALPHA)
        axs[3].plot(time, lo,  color=color, lw=0.8, linestyle='--', alpha=0.6)
        axs[3].plot(time, hi,  color=color, lw=0.8, linestyle='--', alpha=0.6)
        axs[3].plot(time, mid, color=color, lw=1.4, label=lbl)
    label_plot(axs[3], r'Temperatura' + '\n' + r'($T$) [$^\circ$C]', config)
    axs[3].legend(loc='upper left', frameon=True, fontsize=7)
    axs[3].set_xlim(0, time[-1]); axs[3].set_ylim(28, 35)

    lo  = np.cumsum(df_minus['P_cooling'].to_numpy()) * dt / 1000
    mid = np.cumsum(df_base['P_cooling'].to_numpy())  * dt / 1000
    hi  = np.cumsum(df_plus['P_cooling'].to_numpy())  * dt / 1000
    axs[4].fill_between(time, lo, hi, color='seagreen', alpha=ALPHA, label='±20%')
    axs[4].plot(time, lo,  color='seagreen', lw=0.8, linestyle='--', alpha=0.6)
    axs[4].plot(time, hi,  color='seagreen', lw=0.8, linestyle='--', alpha=0.6)
    axs[4].plot(time, mid, color='black',    lw=1.4, label='Base')
    if config == 'vertical':
        axs[4].set_xlabel('Tiempo (s)')
    label_plot(axs[4], 'Energia de Enf.' + '\n' + r'($P_{cool}$) [kJ]', config)
    axs[4].set_xlim(0, time[-1]); axs[4].set_ylim(0, 400)

    _save(name, f'sensitivity_{param_name}')
    plt.show()


def show_results(
    states_hist: np.ndarray | None = None,
    ctrl_hist:   np.ndarray | None = None,
    diag_hist:   np.ndarray | None = None,
    controller_name: str = 'any',
    df: pd.DataFrame | None = None,
    config: str = 'vertical',
):
    if df is None:
        df = _hist_to_df({
            'state': states_hist,
            'controls': ctrl_hist,
            'diagnostics': diag_hist,
        })

    print(f"Total Energy: {df['P_cooling'].sum()/1000:.4f} kJ")
    print(f"Final T_batt: \n{df[['time','T_batt']].tail(3)}")
    plot_results(df, controller_name, config)


def show_sensitivity(
    controller_name: str,
    param_name: str,
    config: str,
    dt: float = 1.0,
    hist_minus: dict[str, list] = {},
    hist_base:  dict[str, list] = {}, 
    hist_plus:  dict[str, list] = {},
    df_minus:   pd.DataFrame | None = None,
    df_base:    pd.DataFrame | None = None,
    df_plus:    pd.DataFrame | None = None,
):
    if df_minus is None: df_minus = _hist_to_df(hist_minus)
    if df_base  is None: df_base  = _hist_to_df(hist_base)
    if df_plus  is None: df_plus  = _hist_to_df(hist_plus)

    print(f"> Parameter modified: {param_name}")
    for tag, df in [('-20%', df_minus), ('Base', df_base), ('+20%', df_plus)]:
        print(f"[{tag}] Total Energy: {df['P_cooling'].sum()/1000:.4f} kJ | "
              f"Max T_batt: {df['T_batt'].max():.3f} °C")

    plot_sensitivity(
        df_minus=df_minus, 
        df_base=df_base, 
        df_plus=df_plus,
        param_name=param_name, 
        name=controller_name,
        config=config, 
        dt=dt,
    )


def plot_learning_history(history: dict):
    episodes = np.arange(len(history['ep_rewards']))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 5), sharex=True)
    ax1.plot(episodes, history['ep_energy_kj'], color='dodgerblue')
    ax1.set_ylabel('Energia' + '\n' + r'Consumida [kJ]')
    ax2.plot(episodes, history['ep_avg_temp'], color='red')
    ax2.set_ylabel(r'Promedio $T_{batt}$ [°C]')
    plt.tight_layout()
    plt.show()

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(episodes[5:], history['ep_rewards'][5:], color='seagreen')
    ax.set_ylabel('Recompensa' + '\n' + r'Cumulativa ($R$)')
    ax.set_xlabel('Episodio')
    ax.ticklabel_format(axis='y')
    plt.tight_layout()
    plt.show()