"""
Simulation de mobilité avec effet Doppler pour scénarios airsoft
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import matplotlib.pyplot as plt
from src.simulator import RadioSimulator, RadioConfig
from src.propagation import Environment
from src.doppler import AIRSOFT_SCENARIOS, velocity_to_kmh, DopplerChannel


def simulate_mobile_scenario(scenario_name: str, distance_m: float = 200):
    """
    Simule un scénario de mobilité
    
    Args:
        scenario_name: Nom du scénario ('marche', 'course', 'vehicule', etc.)
        distance_m: Distance initiale
    """
    scenario = AIRSOFT_SCENARIOS[scenario_name]
    scenario.distance_m = distance_m
    
    config = RadioConfig(
        use_dsss=True,
        encryption_enabled=True,
        environment=Environment.OUTDOOR_OPEN,
        mobile_scenario=scenario,
        enable_rayleigh_fading=False
    )
    
    sim = RadioSimulator(config)
    
    # Génère un message test (32 bits)
    packet_size_bits = 32
    
    print(f"\n=== Scénario: {scenario_name.upper()} ===")
    print(f"Vitesse: {velocity_to_kmh(scenario.velocity_ms):.1f} km/h ({scenario.velocity_ms:.1f} m/s)")
    print(f"Angle: {scenario.angle_deg}°")
    print(f"Distance initiale: {distance_m} m")
    print(f"Durée: {scenario.duration_s} s")
    
    # Calcule le décalage Doppler
    doppler = DopplerChannel(carrier_freq_hz=config.frequency_mhz * 1e6)
    doppler_shift = doppler.calculate_doppler_shift(scenario.velocity_ms, scenario.angle_deg)
    doppler_spread = doppler.calculate_doppler_spread(scenario.velocity_ms)
    
    print(f"\nEffet Doppler:")
    print(f"  Décalage fréquence: {doppler_shift:+.2f} Hz")
    print(f"  Étalement Doppler: {doppler_spread:.2f} Hz")
    
    # Simule à différents instants
    time_points = np.linspace(0, scenario.duration_s, 10)
    bers = []
    distances = []
    snrs = []
    
    for t in time_points:
        # Position à l'instant t
        angle_rad = np.deg2rad(scenario.angle_deg)
        vx = scenario.velocity_ms * np.cos(angle_rad)
        vy = scenario.velocity_ms * np.sin(angle_rad)
        
        x0 = distance_m * np.cos(angle_rad)
        y0 = distance_m * np.sin(angle_rad)
        
        x = x0 + vx * t
        y = y0 + vy * t
        
        current_dist = np.sqrt(x**2 + y**2)
        distances.append(current_dist)
        
        # Simule la transmission
        result = sim.simulate_transmission(packet_size_bits, current_dist, num_packets=10, time_offset_s=t)
        bers.append(result.ber)
        snrs.append(result.snr_db)
    
    print(f"\nRésultats:")
    print(f"  Distance finale: {distances[-1]:.1f} m")
    print(f"  BER moyen: {np.mean(bers):.6f}")
    print(f"  SNR moyen: {np.mean(snrs):.1f} dB")
    
    return time_points, distances, bers, snrs, doppler_shift


def compare_scenarios():
    """Compare différents scénarios de mobilité"""
    
    scenarios_to_test = ['stationnaire', 'marche', 'course', 'approche_rapide', 'eloignement']
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Comparaison des Scénarios de Mobilité - Effet Doppler', fontsize=16, fontweight='bold')
    
    results = {}
    
    for scenario_name in scenarios_to_test:
        t, dist, ber, snr, doppler = simulate_mobile_scenario(scenario_name, distance_m=200)
        results[scenario_name] = (t, dist, ber, snr, doppler)
    
    # Plot 1: Distance vs temps
    ax1 = axes[0, 0]
    for scenario_name, (t, dist, _, _, _) in results.items():
        ax1.plot(t, dist, marker='o', label=scenario_name, linewidth=2)
    ax1.set_xlabel('Temps (s)', fontsize=11)
    ax1.set_ylabel('Distance (m)', fontsize=11)
    ax1.set_title('Évolution de la Distance', fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Plot 2: BER vs temps
    ax2 = axes[0, 1]
    for scenario_name, (t, _, ber, _, _) in results.items():
        ax2.semilogy(t, np.array(ber) + 1e-10, marker='s', label=scenario_name, linewidth=2)
    ax2.set_xlabel('Temps (s)', fontsize=11)
    ax2.set_ylabel('BER', fontsize=11)
    ax2.set_title('Taux d\'Erreur Binaire', fontweight='bold')
    ax2.grid(True, alpha=0.3, which='both')
    ax2.legend()
    
    # Plot 3: SNR vs temps
    ax3 = axes[1, 0]
    for scenario_name, (t, _, _, snr, _) in results.items():
        ax3.plot(t, snr, marker='^', label=scenario_name, linewidth=2)
    ax3.set_xlabel('Temps (s)', fontsize=11)
    ax3.set_ylabel('SNR (dB)', fontsize=11)
    ax3.set_title('Rapport Signal/Bruit', fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    
    # Plot 4: Décalages Doppler
    ax4 = axes[1, 1]
    scenario_labels = []
    doppler_values = []
    velocities = []
    
    for scenario_name in scenarios_to_test:
        scenario = AIRSOFT_SCENARIOS[scenario_name]
        doppler = DopplerChannel(carrier_freq_hz=915e6)
        shift = doppler.calculate_doppler_shift(scenario.velocity_ms, scenario.angle_deg)
        
        scenario_labels.append(scenario_name)
        doppler_values.append(shift)
        velocities.append(velocity_to_kmh(scenario.velocity_ms))
    
    colors = ['green' if d >= 0 else 'red' for d in doppler_values]
    bars = ax4.barh(scenario_labels, doppler_values, color=colors, alpha=0.7, edgecolor='black')
    ax4.set_xlabel('Décalage Doppler (Hz)', fontsize=11)
    ax4.set_title('Décalage de Fréquence par Scénario', fontweight='bold')
    ax4.axvline(0, color='black', linewidth=1, linestyle='--')
    ax4.grid(True, alpha=0.3, axis='x')
    
    # Ajoute les vitesses sur les barres
    for i, (bar, vel) in enumerate(zip(bars, velocities)):
        width = bar.get_width()
        ax4.text(width, bar.get_y() + bar.get_height()/2, 
                f' {vel:.0f} km/h', 
                ha='left' if width >= 0 else 'right',
                va='center', fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('mobility_comparison.png', dpi=300, bbox_inches='tight')
    print(f"\n✓ Graphique sauvegardé: mobility_comparison.png")
    plt.show()


def analyze_velocity_impact():
    """Analyse l'impact de la vitesse sur les performances"""
    
    velocities_kmh = [0, 5, 10, 15, 20, 30, 50, 80, 100]
    distances = [100, 200, 300, 500, 800]
    packet_size_bits = 32
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Impact de la Vitesse sur les Communications Radio', fontsize=16, fontweight='bold')
    
    # Pour chaque distance
    for distance in distances:
        bers = []
        
        for v_kmh in velocities_kmh:
            # Crée un scénario avec cette vitesse
            from src.doppler import MobileScenario, velocity_from_kmh
            
            scenario = MobileScenario(
                velocity_ms=velocity_from_kmh(v_kmh),
                angle_deg=0,  # Directement vers l'émetteur
                distance_m=distance,
                duration_s=10
            )
            
            config = RadioConfig(
                use_dsss=True,
                encryption_enabled=True,
                environment=Environment.OUTDOOR_OPEN,
                mobile_scenario=scenario,
                enable_rayleigh_fading=False
            )
            
            sim = RadioSimulator(config)
            
            # Simule
            result = sim.simulate_transmission(packet_size_bits, distance, num_packets=10)
            bers.append(result.ber)
        
        axes[0].semilogy(velocities_kmh, np.array(bers) + 1e-10, 
                        marker='o', label=f'{distance} m', linewidth=2)
    
    axes[0].set_xlabel('Vitesse (km/h)', fontsize=12)
    axes[0].set_ylabel('BER', fontsize=12)
    axes[0].set_title('BER en fonction de la Vitesse', fontweight='bold')
    axes[0].grid(True, alpha=0.3, which='both')
    axes[0].legend(title='Distance')
    
    # Décalage Doppler vs vitesse
    from src.doppler import velocity_from_kmh
    doppler = DopplerChannel(carrier_freq_hz=915e6)
    doppler_shifts = [doppler.calculate_doppler_shift(velocity_from_kmh(v), 0) for v in velocities_kmh]
    doppler_spreads = [doppler.calculate_doppler_spread(velocity_from_kmh(v)) for v in velocities_kmh]
    
    axes[1].plot(velocities_kmh, doppler_shifts, marker='o', label='Décalage (0°)', linewidth=2, color='blue')
    axes[1].plot(velocities_kmh, doppler_spreads, marker='s', label='Étalement', linewidth=2, color='red')
    axes[1].set_xlabel('Vitesse (km/h)', fontsize=12)
    axes[1].set_ylabel('Fréquence (Hz)', fontsize=12)
    axes[1].set_title('Effet Doppler vs Vitesse', fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    
    plt.tight_layout()
    plt.savefig('velocity_impact.png', dpi=300, bbox_inches='tight')
    print(f"✓ Graphique sauvegardé: velocity_impact.png")
    plt.show()


if __name__ == "__main__":
    print("=" * 70)
    print("SIMULATION DE MOBILITÉ - EFFET DOPPLER")
    print("Système Radio Airsoft avec DSSS et Chiffrement AES-256")
    print("=" * 70)
    
    # Comparaison des scénarios
    print("\n--- Comparaison des Scénarios ---")
    compare_scenarios()
    
    # Impact de la vitesse
    print("\n--- Impact de la Vitesse ---")
    analyze_velocity_impact()
    
    print("\n" + "=" * 70)
    print("✓ Simulations de mobilité terminées!")
    print("=" * 70)
