"""
Démonstration du système adaptatif en temps réel
Simulation d'un scénario airsoft avec conditions variables
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import matplotlib.pyplot as plt
from src.simulator import RadioSimulator, RadioConfig
from src.propagation import Environment
from src.doppler import MobileScenario, velocity_from_kmh, DopplerChannel
from src.adaptive import AdaptiveController, AdaptiveMode, ChannelEstimate


def simulate_adaptive_scenario():
    """
    Simule un scénario avec conditions variables et adaptation temps réel
    """
    print("=" * 70)
    print("SIMULATION ADAPTATIVE - SCÉNARIO AIRSOFT RÉALISTE")
    print("=" * 70)
    
    # Configuration initiale
    config = RadioConfig(
        use_dsss=True,
        chip_rate=1000000,
        data_rate=10000,
        tx_power_dbm=20.0,
        encryption_enabled=True,
        environment=Environment.OUTDOOR_OPEN
    )
    
    # Contrôleur adaptatif
    controller = AdaptiveController(mode=AdaptiveMode.FULL_AUTO)
    
    # Scénario : Joueur en mouvement sur terrain avec obstacles
    # Phase 1: Terrain dégagé (0-5s)
    # Phase 2: Entrée en forêt dense (5-10s) 
    # Phase 3: Course rapide (10-15s)
    # Phase 4: Ralentissement, zone urbaine (15-20s)
    
    time_points = np.linspace(0, 20, 40)  # 40 points sur 20 secondes
    distances = []
    velocities = []
    environments = []
    
    for t in time_points:
        if t < 5:
            # Phase 1: Marche lente, terrain dégagé
            distance = 200 + 1.4 * t  # 5 km/h
            velocity = 1.4
            env = Environment.OUTDOOR_OPEN
        elif t < 10:
            # Phase 2: Marche en forêt (plus de bruit)
            distance = 200 + 1.4 * 5 + 1.4 * (t - 5)
            velocity = 1.4
            env = Environment.OUTDOOR_SUBURBAN  # Simule végétation
        elif t < 15:
            # Phase 3: Course rapide
            distance = 200 + 1.4 * 10 + 4.2 * (t - 10)  # 15 km/h
            velocity = 4.2
            env = Environment.OUTDOOR_SUBURBAN
        else:
            # Phase 4: Zone urbaine, ralentissement
            distance = 200 + 1.4 * 10 + 4.2 * 5 + 2.8 * (t - 15)  # 10 km/h
            velocity = 2.8
            env = Environment.OUTDOOR_URBAN
        
        distances.append(distance)
        velocities.append(velocity)
        environments.append(env)
    
    # Simulations avec et sans adaptation
    results_adaptive = []
    results_fixed = []
    
    print("\nSimulation en cours...")
    print(f"{'Temps':>6} {'Phase':>12} {'Dist':>7} {'Vit':>6} {'SNR':>7} {'BER':>10} {'Adapt':>30}")
    print("-" * 95)
    
    for i, (t, dist, vel, env) in enumerate(zip(time_points, distances, velocities, environments)):
        # Détermine la phase
        if t < 5:
            phase = "Dégagé"
        elif t < 10:
            phase = "Forêt"
        elif t < 15:
            phase = "Course"
        else:
            phase = "Urbain"
        
        # ===== SYSTÈME ADAPTATIF =====
        # Crée un scénario mobile
        scenario = MobileScenario(
            velocity_ms=vel,
            angle_deg=0,
            distance_m=dist,
            duration_s=0.5
        )
        
        config_adapt = RadioConfig(
            use_dsss=controller.params.use_dsss,
            chip_rate=1000000,
            data_rate=controller.params.data_rate,
            tx_power_dbm=controller.params.tx_power_dbm,
            encryption_enabled=True,
            environment=env,
            mobile_scenario=scenario
        )
        
        sim_adapt = RadioSimulator(config_adapt)
        result_adapt = sim_adapt.simulate_transmission(32, dist, num_packets=5, time_offset_s=0)
        
        # Estimation du canal pour adapter
        signal_tx = np.random.randint(0, 2, size=320)
        signal_rx = signal_tx.copy()
        num_errors = int(len(signal_rx) * result_adapt.ber)
        if num_errors > 0:
            error_idx = np.random.choice(len(signal_rx), num_errors, replace=False)
            signal_rx[error_idx] = 1 - signal_rx[error_idx]
        
        # Calcul du Doppler
        doppler = DopplerChannel(carrier_freq_hz=915e6)
        doppler_shift = doppler.calculate_doppler_shift(vel, 0)
        doppler_spread = doppler.calculate_doppler_spread(vel)
        
        channel_est = ChannelEstimate(
            snr_db=result_adapt.snr_db,
            doppler_shift_hz=doppler_shift,
            doppler_spread_hz=doppler_spread,
            ber=result_adapt.ber,
            per=result_adapt.per,
            rx_power_dbm=result_adapt.rx_power_dbm,
            noise_floor_dbm=-174 + 10 * np.log10(config_adapt.data_rate)
        )
        
        # Adaptation des paramètres
        new_params = controller.update(channel_est)
        
        results_adaptive.append({
            'time': t,
            'distance': dist,
            'velocity': vel,
            'environment': env.name,
            'snr_db': result_adapt.snr_db,
            'ber': result_adapt.ber,
            'per': result_adapt.per,
            'tx_power': new_params.tx_power_dbm,
            'data_rate': new_params.data_rate,
            'sf': new_params.spreading_factor,
            'doppler': doppler_shift
        })
        
        # ===== SYSTÈME FIXE (pour comparaison) =====
        config_fixed = RadioConfig(
            use_dsss=True,
            chip_rate=1000000,
            data_rate=10000,
            tx_power_dbm=20.0,
            encryption_enabled=True,
            environment=env,
            mobile_scenario=scenario
        )
        
        sim_fixed = RadioSimulator(config_fixed)
        result_fixed = sim_fixed.simulate_transmission(32, dist, num_packets=5, time_offset_s=0)
        
        results_fixed.append({
            'time': t,
            'snr_db': result_fixed.snr_db,
            'ber': result_fixed.ber,
            'per': result_fixed.per
        })
        
        # Affichage
        adapt_info = f"Pwr={new_params.tx_power_dbm:.0f}dBm Rate={new_params.data_rate//1000}k SF={new_params.spreading_factor}"
        print(f"{t:>6.1f} {phase:>12} {dist:>7.1f} {vel*3.6:>6.1f} {result_adapt.snr_db:>7.1f} "
              f"{result_adapt.ber:>10.2e} {adapt_info:>30}")
    
    print("\n✓ Simulation terminée")
    
    # Affiche les statistiques
    print("\n" + "=" * 70)
    print("STATISTIQUES")
    print("=" * 70)
    
    stats = controller.get_statistics()
    print(f"\nAdaptations effectuées: {stats['num_adaptations']}")
    print(f"SNR moyen: {stats['avg_snr_db']:.1f} dB")
    print(f"BER moyen: {stats['avg_ber']:.2e}")
    print(f"Doppler moyen: {stats['avg_doppler_hz']:.1f} Hz")
    print(f"Changements de puissance: {stats.get('power_changes', 0)}")
    print(f"Changements de débit: {stats.get('rate_changes', 0)}")
    
    # Comparaison Adaptatif vs Fixe
    ber_adapt_avg = np.mean([r['ber'] for r in results_adaptive])
    ber_fixed_avg = np.mean([r['ber'] for r in results_fixed])
    per_adapt_avg = np.mean([r['per'] for r in results_adaptive])
    per_fixed_avg = np.mean([r['per'] for r in results_fixed])
    
    print(f"\n--- Comparaison Adaptatif vs Fixe ---")
    print(f"BER - Adaptatif: {ber_adapt_avg:.2e} | Fixe: {ber_fixed_avg:.2e}")
    print(f"PER - Adaptatif: {per_adapt_avg:.2%} | Fixe: {per_fixed_avg:.2%}")
    
    improvement_ber = (ber_fixed_avg - ber_adapt_avg) / ber_fixed_avg * 100 if ber_fixed_avg > 0 else 0
    improvement_per = (per_fixed_avg - per_adapt_avg) / per_fixed_avg * 100 if per_fixed_avg > 0 else 0
    
    print(f"Amélioration BER: {improvement_ber:+.1f}%")
    print(f"Amélioration PER: {improvement_per:+.1f}%")
    
    return results_adaptive, results_fixed


def plot_adaptive_results(results_adaptive, results_fixed):
    """Génère les graphiques de comparaison"""
    
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(4, 2, hspace=0.3, wspace=0.3)
    
    times = [r['time'] for r in results_adaptive]
    
    # Plot 1: Distance et vitesse
    ax1 = fig.add_subplot(gs[0, 0])
    ax1_twin = ax1.twinx()
    
    distances = [r['distance'] for r in results_adaptive]
    velocities = [r['velocity'] * 3.6 for r in results_adaptive]  # km/h
    
    ax1.plot(times, distances, 'b-', linewidth=2, label='Distance')
    ax1_twin.plot(times, velocities, 'r--', linewidth=2, label='Vitesse')
    
    ax1.set_xlabel('Temps (s)', fontsize=11)
    ax1.set_ylabel('Distance (m)', fontsize=11, color='b')
    ax1_twin.set_ylabel('Vitesse (km/h)', fontsize=11, color='r')
    ax1.set_title('Scénario de Mobilité', fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(axis='y', labelcolor='b')
    ax1_twin.tick_params(axis='y', labelcolor='r')
    
    # Zones de phase
    ax1.axvspan(0, 5, alpha=0.1, color='green', label='Dégagé')
    ax1.axvspan(5, 10, alpha=0.1, color='yellow', label='Forêt')
    ax1.axvspan(10, 15, alpha=0.1, color='orange', label='Course')
    ax1.axvspan(15, 20, alpha=0.1, color='red', label='Urbain')
    
    # Plot 2: SNR Comparaison
    ax2 = fig.add_subplot(gs[0, 1])
    snr_adapt = [r['snr_db'] for r in results_adaptive]
    snr_fixed = [r['snr_db'] for r in results_fixed]
    
    ax2.plot(times, snr_adapt, 'g-', linewidth=2, marker='o', label='Adaptatif', markersize=4)
    ax2.plot(times, snr_fixed, 'r--', linewidth=2, marker='s', label='Fixe', markersize=4)
    ax2.axhline(15, color='orange', linestyle=':', linewidth=1, label='Seuil dégradé')
    ax2.axhline(10, color='red', linestyle=':', linewidth=1, label='Seuil critique')
    
    ax2.set_xlabel('Temps (s)', fontsize=11)
    ax2.set_ylabel('SNR (dB)', fontsize=11)
    ax2.set_title('Rapport Signal/Bruit', fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # Plot 3: BER Comparaison
    ax3 = fig.add_subplot(gs[1, 0])
    ber_adapt = [r['ber'] + 1e-10 for r in results_adaptive]
    ber_fixed = [r['ber'] + 1e-10 for r in results_fixed]
    
    ax3.semilogy(times, ber_adapt, 'g-', linewidth=2, marker='o', label='Adaptatif', markersize=4)
    ax3.semilogy(times, ber_fixed, 'r--', linewidth=2, marker='s', label='Fixe', markersize=4)
    ax3.axhline(1e-3, color='orange', linestyle=':', linewidth=1, label='Cible')
    
    ax3.set_xlabel('Temps (s)', fontsize=11)
    ax3.set_ylabel('BER', fontsize=11)
    ax3.set_title('Taux d\'Erreur Binaire', fontweight='bold')
    ax3.grid(True, alpha=0.3, which='both')
    ax3.legend()
    
    # Plot 4: PER Comparaison
    ax4 = fig.add_subplot(gs[1, 1])
    per_adapt = [r['per'] * 100 for r in results_adaptive]
    per_fixed = [r['per'] * 100 for r in results_fixed]
    
    ax4.plot(times, per_adapt, 'g-', linewidth=2, marker='o', label='Adaptatif', markersize=4)
    ax4.plot(times, per_fixed, 'r--', linewidth=2, marker='s', label='Fixe', markersize=4)
    
    ax4.set_xlabel('Temps (s)', fontsize=11)
    ax4.set_ylabel('PER (%)', fontsize=11)
    ax4.set_title('Taux d\'Erreur de Paquets', fontweight='bold')
    ax4.grid(True, alpha=0.3)
    ax4.legend()
    
    # Plot 5: Adaptation de la puissance TX
    ax5 = fig.add_subplot(gs[2, 0])
    tx_power = [r['tx_power'] for r in results_adaptive]
    
    ax5.plot(times, tx_power, 'b-', linewidth=2, marker='o', markersize=5)
    ax5.axhline(20, color='gray', linestyle='--', linewidth=1, label='Puissance initiale')
    ax5.fill_between(times, 0, 27, alpha=0.1, color='green', label='Plage légale')
    
    ax5.set_xlabel('Temps (s)', fontsize=11)
    ax5.set_ylabel('Puissance TX (dBm)', fontsize=11)
    ax5.set_title('Adaptation de Puissance', fontweight='bold')
    ax5.grid(True, alpha=0.3)
    ax5.set_ylim(0, 30)
    ax5.legend()
    
    # Plot 6: Adaptation du débit
    ax6 = fig.add_subplot(gs[2, 1])
    data_rates = [r['data_rate'] / 1000 for r in results_adaptive]
    
    ax6.step(times, data_rates, 'purple', linewidth=2, where='post')
    ax6.axhline(10, color='gray', linestyle='--', linewidth=1, label='Débit initial')
    
    ax6.set_xlabel('Temps (s)', fontsize=11)
    ax6.set_ylabel('Débit (kbps)', fontsize=11)
    ax6.set_title('Adaptation du Débit', fontweight='bold')
    ax6.grid(True, alpha=0.3)
    ax6.legend()
    
    # Plot 7: Spreading Factor
    ax7 = fig.add_subplot(gs[3, 0])
    spreading_factors = [r['sf'] for r in results_adaptive]
    
    ax7.step(times, spreading_factors, 'orange', linewidth=2, where='post')
    ax7.axhline(100, color='gray', linestyle='--', linewidth=1, label='SF initial')
    
    ax7.set_xlabel('Temps (s)', fontsize=11)
    ax7.set_ylabel('Spreading Factor', fontsize=11)
    ax7.set_title('Adaptation du Facteur d\'Étalement', fontweight='bold')
    ax7.grid(True, alpha=0.3)
    ax7.legend()
    
    # Plot 8: Effet Doppler
    ax8 = fig.add_subplot(gs[3, 1])
    doppler_shifts = [r['doppler'] for r in results_adaptive]
    
    ax8.plot(times, doppler_shifts, 'cyan', linewidth=2, marker='o', markersize=4)
    ax8.axhline(0, color='black', linestyle='-', linewidth=1)
    ax8.fill_between(times, -50, 50, alpha=0.1, color='green', label='Impact faible')
    
    ax8.set_xlabel('Temps (s)', fontsize=11)
    ax8.set_ylabel('Décalage Doppler (Hz)', fontsize=11)
    ax8.set_title('Effet Doppler Instantané', fontweight='bold')
    ax8.grid(True, alpha=0.3)
    ax8.legend()
    
    plt.suptitle('Système Radio Adaptatif - Scénario Airsoft Complet', 
                fontsize=16, fontweight='bold', y=0.995)
    
    plt.savefig('adaptive_scenario.png', dpi=300, bbox_inches='tight')
    print("\n✓ Graphique sauvegardé: adaptive_scenario.png")
    plt.show()


if __name__ == "__main__":
    # Exécute la simulation
    results_adaptive, results_fixed = simulate_adaptive_scenario()
    
    # Génère les graphiques
    print("\n--- Génération des graphiques ---")
    plot_adaptive_results(results_adaptive, results_fixed)
    
    print("\n" + "=" * 70)
    print("✓ Simulation adaptative terminée avec succès!")
    print("=" * 70)
