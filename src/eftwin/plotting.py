"""Plotting helpers separated from the core algorithms."""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from .constants import TOWER_HEIGHTS_9
from .modal_analysis import modal_parameters, acceleration_mode_to_displacement


def plot_eigs_custom(analysis_result, dt=0.0125, font_size=14, save_path=None):
    hdmd = analysis_result.model if hasattr(analysis_result, "model") else analysis_result["model"]
    name = analysis_result.name if hasattr(analysis_result, "name") else analysis_result.get("name", "DMD")
    eigs = hdmd.eigs
    _, freqs, _ = modal_parameters(eigs, dt)
    highlight = ((freqs > 0.4) & (freqs < 0.6)) | ((freqs > 1.5) & (freqs < 2.5))
    fig, ax = plt.subplots(figsize=(8, 8))
    theta = np.linspace(0, 2*np.pi, 200)
    ax.plot(np.cos(theta), np.sin(theta), 'k--', linewidth=1.5, alpha=0.6, label='Stability Limit')
    ax.scatter(eigs[~highlight].real, eigs[~highlight].imag, s=60, alpha=0.6, label='Other Modes', edgecolors='k', linewidth=0.5)
    ax.scatter(eigs[highlight].real, eigs[highlight].imag, s=150, marker='*', label='Tower Candidate Modes', zorder=10, edgecolors='k')
    ax.set_title(f"{name}: Stability Map (Eigenvalues)", fontsize=font_size+2, weight='bold')
    ax.set_xlabel("Real Part", fontsize=font_size)
    ax.set_ylabel("Imaginary Part", fontsize=font_size)
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.legend(fontsize=12, loc='lower left')
    ax.set_aspect('equal')
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.2, 1.2)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    return fig, ax


def plot_mode_shapes(analysis_result, dt=0.0125, low_f=0.3, high_f=5.0, max_modes=10, save_path=None):
    hdmd = analysis_result.model if hasattr(analysis_result, "model") else analysis_result["model"]
    name = analysis_result.name if hasattr(analysis_result, "name") else analysis_result.get("name", "DMD")
    _, freqs, _ = modal_parameters(hdmd.eigs, dt)
    idxs = np.where((freqs > low_f) & (freqs < high_f))[0]
    idxs = idxs[np.argsort(freqs[idxs])]
    unique = []
    fig, ax = plt.subplots(figsize=(10, 7))
    plot_count = 0
    for idx in idxs:
        f = freqs[idx]
        if any(abs(f-u) < 0.1 for u in unique):
            continue
        unique.append(f)
        phi = hdmd.modes[:, idx][0:18]
        disp = acceleration_mode_to_displacement(phi[0:9], f)
        max_val = disp[np.argmax(np.abs(disp))]
        disp_norm = disp / max_val if max_val != 0 else disp
        if plot_count < max_modes:
            ax.plot(disp_norm.real, TOWER_HEIGHTS_9, marker='s' if f > 1.0 else 'o', linestyle='--' if f > 1.0 else '-', linewidth=2, label=f'{f:.2f} Hz')
            plot_count += 1
    ax.set_title(f"{name}: DMD Mode Shapes", fontweight='bold')
    ax.set_xlabel("Normalized Displacement")
    ax.set_ylabel("Tower Height (m)")
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.5)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    return fig, ax


def plot_virtual_sensing_result(result, sensor_index, plot_duration=None, save_path=None):
    pred = result['pred']
    truth = result['truth']
    dt = result.get('dt', 0.0125)
    total_steps = pred.shape[1]
    steps = min(int(plot_duration/dt), total_steps) if plot_duration else total_steps
    t = np.arange(steps) * dt
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(t, truth[sensor_index, :steps], 'k-', alpha=0.5, label='Actual (Hidden)')
    ax.plot(t, pred[sensor_index, :steps], 'r--', linewidth=1.2, label='Virtual Sensing (Recovered)')
    ax.set_title(f"Virtual Sensing Recovery: Sensor Index {sensor_index}")
    ax.set_ylabel("Value")
    ax.set_xlabel("Time (s)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    return fig, ax


def plot_stabilization_diagram(poles, title="SSI Stabilization Diagram", save_path=None):
    fig, ax = plt.subplots(figsize=(12, 8))
    color_map = {'stable': 'green', 'stable_freq_mac': 'blue', 'stable_freq': 'cyan', 'new': 'grey'}
    size_map = {'stable': 30, 'stable_freq_mac': 20, 'stable_freq': 10, 'new': 5}
    for status in ['new', 'stable_freq', 'stable_freq_mac', 'stable']:
        subset = [p for p in poles if p['status'] == status]
        if subset:
            ax.scatter([p['f'] for p in subset], [p['order'] for p in subset], c=color_map[status], s=size_map[status], alpha=0.7, label=status)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Model Order")
    ax.set_title(title)
    ax.set_xlim(0, 5.0)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    return fig, ax


def plot_mac_heatmap(mac_matrix, title="Mode Shape Validation", save_path=None):
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.heatmap(mac_matrix, annot=True, fmt='.2f', cmap='Blues', vmin=0, vmax=1, ax=ax)
    ax.set_title(title)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    return fig, ax


def plot_delay_sensitivity(sensitivity_df, title="Hankel Delay Sensitivity", convergence_window=(45, 75), selected_d=60, save_path=None):
    """Twin-axis plot of frequency and damping vs Hankel delay d.

    Reproduces the layout used in the original ``Probablistic_Mode_shape 2.py``.
    """
    fig, ax1 = plt.subplots(figsize=(10, 6))
    color_freq = 'tab:blue'
    ax1.set_xlabel('Hankel Delay (d)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Frequency (Hz)', color=color_freq, fontsize=12, fontweight='bold')
    ax1.plot(sensitivity_df['d'], sensitivity_df['Freq'], color=color_freq, marker='o', linewidth=2, label='Frequency')
    ax1.tick_params(axis='y', labelcolor=color_freq)
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    color_damp = 'tab:red'
    ax2.set_ylabel('Damping Ratio', color=color_damp, fontsize=12, fontweight='bold')
    ax2.plot(sensitivity_df['d'], sensitivity_df['Damping'], color=color_damp, marker='s', linestyle='--', linewidth=2, label='Damping')
    ax2.tick_params(axis='y', labelcolor=color_damp)

    ax1.axvspan(convergence_window[0], convergence_window[1], color='gray', alpha=0.15, label=f'Convergence Zone d=[{convergence_window[0]},{convergence_window[1]}]')
    ax1.axvline(selected_d, color='k', linestyle=':', linewidth=2, label=f'Selected d={selected_d}')

    ax1.set_title(title, fontsize=14)
    fig.legend(loc='upper right', bbox_to_anchor=(0.92, 0.92), frameon=True)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    return fig, (ax1, ax2)


def plot_probabilistic_mode_shape(prob_result, tower_heights, title=None, save_path=None):
    """Plot the mean mode shape with a 95% confidence band.

    Expects the dictionary returned by ``probabilistic_modes.run_probabilistic_mode_sweep``.
    """
    shapes = prob_result['shapes']
    mean_shape = prob_result['mean_shape']
    ci_lower = prob_result['ci_lower']
    ci_upper = prob_result['ci_upper']
    direction_name = prob_result.get('direction_name', '')

    fig, ax = plt.subplots(figsize=(8, 10))
    for s in shapes:
        ax.plot(s, tower_heights, color='gray', alpha=0.3, linewidth=1)
    ax.fill_betweenx(tower_heights, ci_lower, ci_upper, alpha=0.2, label='95% Confidence (Variance due to d)')
    ax.plot(mean_shape, tower_heights, marker='o', linewidth=3, label='Mean Mode Shape')
    cw = prob_result.get('convergence_window', (None, None))
    plot_title = title or f"{direction_name}: Probabilistic Mode Shape\n(Derived from Convergence Zone d=[{cw[0]},{cw[1]}])"
    ax.set_title(plot_title, fontsize=14)
    ax.set_xlabel('Normalized Displacement', fontsize=12)
    ax.set_ylabel('Tower Height (m)', fontsize=12)
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.5)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    return fig, ax
