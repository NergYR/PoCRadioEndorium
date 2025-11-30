"""
Visualisation de l'effet Doppler sur le spectre et la constellation
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import matplotlib.pyplot as plt
from src.doppler import DopplerChannel, AIRSOFT_SCENARIOS, velocity_to_kmh
from src.modulation import DSSSModulator


def visualize_doppler_spectrum():
    """Visualise l'effet Doppler sur le spectre du signal"""
    
    # Paramètres
    carrier_freq = 915e6
    sample_rate = 1e6
    signal_duration = 0.01  # 10 ms
    
    doppler = DopplerChannel(carrier_freq_hz=carrier_freq)
    
    # Génère un signal DSSS test
    dsss = DSSSModulator(chip_rate=int(sample_rate), data_rate=10000)
    test_data = np.random.randint(0, 2, size=100)
    signal = dsss.spread(test_data)
    
    # Tronque à la durée voulue
    n_samples = int(signal_duration * sample_rate)
    signal = signal[:n_samples]
    
    # Crée une figure avec 6 subplots
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
    
    scenarios_to_plot = ['stationnaire', 'course', 'vehicule']
    colors = ['green', 'orange', 'red']
    
    # Plot 1: Spectres superposés
    ax1 = fig.add_subplot(gs[0, :])
    
    for scenario_name, color in zip(scenarios_to_plot, colors):
        scenario = AIRSOFT_SCENARIOS[scenario_name]
        doppler_shift = doppler.calculate_doppler_shift(scenario.velocity_ms, scenario.angle_deg)
        
        # Applique le Doppler
        shifted_signal = doppler.apply_doppler_to_signal(
            signal.astype(complex), sample_rate, doppler_shift
        )
        
        # FFT
        fft = np.fft.fftshift(np.fft.fft(shifted_signal))
        freqs = np.fft.fftshift(np.fft.fftfreq(len(shifted_signal), 1/sample_rate))
        
        # Plot du spectre
        power_db = 20 * np.log10(np.abs(fft) + 1e-10)
        ax1.plot(freqs/1e3, power_db, label=f"{scenario_name} ({doppler_shift:+.1f} Hz)", 
                color=color, linewidth=2, alpha=0.7)
    
    ax1.set_xlabel('Fréquence (kHz)', fontsize=12)
    ax1.set_ylabel('Puissance (dB)', fontsize=12)
    ax1.set_title('Spectres avec Effet Doppler', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    ax1.set_xlim(-100, 100)
    
    # Plots 2-4: Signaux temporels
    for idx, (scenario_name, color) in enumerate(zip(scenarios_to_plot, colors)):
        ax = fig.add_subplot(gs[1, idx % 2])
        
        scenario = AIRSOFT_SCENARIOS[scenario_name]
        doppler_shift = doppler.calculate_doppler_shift(scenario.velocity_ms, scenario.angle_deg)
        
        # Signal avec Doppler
        shifted_signal = doppler.apply_doppler_to_signal(
            signal.astype(complex), sample_rate, doppler_shift
        )
        
        # Affiche une portion du signal
        t_ms = np.arange(min(1000, len(shifted_signal))) / sample_rate * 1000
        ax.plot(t_ms, np.real(shifted_signal[:len(t_ms)]), color=color, linewidth=1.5)
        ax.set_xlabel('Temps (ms)', fontsize=11)
        ax.set_ylabel('Amplitude', fontsize=11)
        ax.set_title(f'{scenario_name.capitalize()} - {velocity_to_kmh(scenario.velocity_ms):.0f} km/h', 
                    fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        if idx >= 2:
            ax = fig.add_subplot(gs[2, (idx - 2) % 2])
            # FFT individuelle
            fft = np.fft.fftshift(np.fft.fft(shifted_signal))
            freqs = np.fft.fftshift(np.fft.fftfreq(len(shifted_signal), 1/sample_rate))
            power_db = 20 * np.log10(np.abs(fft) + 1e-10)
            ax.plot(freqs/1e3, power_db, color=color, linewidth=2)
            ax.set_xlabel('Fréquence (kHz)', fontsize=11)
            ax.set_ylabel('Puissance (dB)', fontsize=11)
            ax.set_title(f'Spectre - {scenario_name}', fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.set_xlim(-100, 100)
    
    # Plot 5: Constellation avec et sans Doppler (course)
    ax5 = fig.add_subplot(gs[2, 0])
    scenario = AIRSOFT_SCENARIOS['course']
    doppler_shift = doppler.calculate_doppler_shift(scenario.velocity_ms, scenario.angle_deg)
    
    # Signal sans Doppler
    ax5.scatter(np.real(signal[:500]), np.imag(signal[:500].astype(complex)), 
               alpha=0.5, s=10, label='Sans Doppler', color='blue')
    
    # Signal avec Doppler
    shifted = doppler.apply_doppler_to_signal(signal.astype(complex), sample_rate, doppler_shift)
    ax5.scatter(np.real(shifted[:500]), np.imag(shifted[:500]), 
               alpha=0.5, s=10, label='Avec Doppler', color='red')
    
    ax5.set_xlabel('I (In-phase)', fontsize=11)
    ax5.set_ylabel('Q (Quadrature)', fontsize=11)
    ax5.set_title('Constellation - Effet Doppler', fontweight='bold')
    ax5.grid(True, alpha=0.3)
    ax5.legend()
    ax5.axis('equal')
    
    # Plot 6: Tableau récapitulatif
    ax6 = fig.add_subplot(gs[2, 1])
    ax6.axis('off')
    
    table_data = []
    table_data.append(['Scénario', 'Vitesse', 'Doppler', 'Étalement'])
    
    for scenario_name in ['stationnaire', 'marche', 'course', 'approche_rapide', 'vehicule']:
        scenario = AIRSOFT_SCENARIOS[scenario_name]
        v_kmh = velocity_to_kmh(scenario.velocity_ms)
        shift = doppler.calculate_doppler_shift(scenario.velocity_ms, scenario.angle_deg)
        spread = doppler.calculate_doppler_spread(scenario.velocity_ms)
        
        table_data.append([
            scenario_name.capitalize(),
            f'{v_kmh:.0f} km/h',
            f'{shift:+.1f} Hz',
            f'{spread:.1f} Hz'
        ])
    
    table = ax6.table(cellText=table_data, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2.5)
    
    # Style du header
    for i in range(4):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Style des lignes alternées
    for i in range(1, len(table_data)):
        for j in range(4):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#f0f0f0')
    
    ax6.set_title('Paramètres Doppler', fontsize=12, fontweight='bold', pad=20)
    
    plt.suptitle('Visualisation de l\'Effet Doppler - Système Radio 915 MHz', 
                fontsize=16, fontweight='bold', y=0.995)
    
    plt.savefig('doppler_spectrum.png', dpi=300, bbox_inches='tight')
    print("✓ Graphique sauvegardé: doppler_spectrum.png")
    plt.show()


def plot_doppler_trajectory():
    """Trace les trajectoires et décalages Doppler"""
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle('Trajectoires et Décalages Doppler Instantanés', fontsize=16, fontweight='bold')
    
    doppler = DopplerChannel(carrier_freq_hz=915e6)
    sample_rate = 1e6
    
    scenarios = ['approche_rapide', 'eloignement', 'marche', 'course', 'vehicule']
    colors_map = {'approche_rapide': 'blue', 'eloignement': 'red', 'marche': 'green', 
                  'course': 'orange', 'vehicule': 'purple'}
    
    for idx, scenario_name in enumerate(scenarios):
        row = idx // 3
        col = idx % 3
        
        scenario = AIRSOFT_SCENARIOS[scenario_name]
        
        # Simule la trajectoire
        t, doppler_shifts, distances = doppler.simulate_mobile_trajectory(scenario, sample_rate)
        
        # Plot trajectoire
        if row == 0:
            axes[row, col].plot(t, distances, color=colors_map[scenario_name], linewidth=2)
            axes[row, col].set_xlabel('Temps (s)', fontsize=10)
            axes[row, col].set_ylabel('Distance (m)', fontsize=10)
            axes[row, col].set_title(f'{scenario_name.replace("_", " ").title()}\n{velocity_to_kmh(scenario.velocity_ms):.0f} km/h',
                                    fontweight='bold')
            axes[row, col].grid(True, alpha=0.3)
        else:
            axes[row, col].plot(t, doppler_shifts, color=colors_map[scenario_name], linewidth=2)
            axes[row, col].set_xlabel('Temps (s)', fontsize=10)
            axes[row, col].set_ylabel('Décalage Doppler (Hz)', fontsize=10)
            axes[row, col].set_title(f'{scenario_name.replace("_", " ").title()}', fontweight='bold')
            axes[row, col].grid(True, alpha=0.3)
            axes[row, col].axhline(0, color='black', linestyle='--', linewidth=1)
    
    # Dernière case : résumé
    axes[1, 2].axis('off')
    summary_text = "Observations:\n\n"
    summary_text += "• Approche: Doppler positif\n"
    summary_text += "• Éloignement: Doppler négatif\n"
    summary_text += "• Doppler ∝ vitesse radiale\n"
    summary_text += "• Doppler max @ 915 MHz:\n"
    summary_text += "  - 50 km/h: ±42 Hz\n"
    summary_text += "  - 100 km/h: ±85 Hz\n"
    
    axes[1, 2].text(0.5, 0.5, summary_text, 
                   ha='center', va='center',
                   fontsize=11, family='monospace',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig('doppler_trajectories.png', dpi=300, bbox_inches='tight')
    print("✓ Graphique sauvegardé: doppler_trajectories.png")
    plt.show()


if __name__ == "__main__":
    print("=" * 70)
    print("VISUALISATION DE L'EFFET DOPPLER")
    print("=" * 70)
    
    print("\n--- Génération du spectre Doppler ---")
    visualize_doppler_spectrum()
    
    print("\n--- Génération des trajectoires ---")
    plot_doppler_trajectory()
    
    print("\n" + "=" * 70)
    print("✓ Visualisations terminées!")
    print("=" * 70)
