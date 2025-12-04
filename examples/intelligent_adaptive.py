"""
Démonstration du système adaptatif intelligent amélioré
Montre l'adaptation automatique avec prédiction, FEC adaptatif, et entrelacement
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.simulator import RadioSimulator, RadioConfig
from src.adaptive import AdaptiveController, AdaptiveMode
from src.propagation import Environment
from src.doppler import MobileScenario


def create_complex_scenario():
    """Crée un scénario complexe avec phases variées"""
    phases = []
    
    # Phase 1: Terrain dégagé, marche lente (5 km/h)
    phases.append({
        'name': 'Patrouille terrain dégagé',
        'duration': 10,
        'environment': Environment.OUTDOOR_OPEN,
        'scenario': MobileScenario(
            velocity_ms=5/3.6,  # 5 km/h
            angle_deg=45,
            distance_m=200
        ),
        'base_noise': -110  # dBm
    })
    
    # Phase 2: Forêt dense, marche normale (7 km/h)
    phases.append({
        'name': 'Progression en forêt',
        'duration': 10,
        'environment': Environment.OUTDOOR_FOREST,
        'scenario': MobileScenario(
            velocity_ms=7/3.6,
            angle_deg=90,
            distance_m=300
        ),
        'base_noise': -108
    })
    
    # Phase 3: Course rapide en zone urbaine (15 km/h)
    phases.append({
        'name': 'Course urbaine',
        'duration': 10,
        'environment': Environment.INDOOR_OFFICE,
        'scenario': MobileScenario(
            velocity_ms=15/3.6,
            angle_deg=120,
            distance_m=150
        ),
        'base_noise': -105
    })
    
    # Phase 4: Ralentissement, obstacles denses (8 km/h)
    phases.append({
        'name': 'CQB intérieur',
        'duration': 10,
        'environment': Environment.INDOOR_FACTORY,
        'scenario': MobileScenario(
            velocity_ms=8/3.6,
            angle_deg=180,
            distance_m=80
        ),
        'base_noise': -106
    })
    
    # Phase 5: Repositionnement rapide (20 km/h)
    phases.append({
        'name': 'Sprint repositionnement',
        'duration': 10,
        'environment': Environment.OUTDOOR_OPEN,
        'scenario': MobileScenario(
            velocity_ms=20/3.6,
            angle_deg=270,
            distance_m=400
        ),
        'base_noise': -112
    })
    
    return phases


def simulate_intelligent_system():
    """Simule le système adaptatif intelligent"""
    
    print("=" * 70)
    print("🎯 DÉMONSTRATION SYSTÈME ADAPTATIF INTELLIGENT")
    print("=" * 70)
    print()
    print("Fonctionnalités avancées:")
    print("  ✓ Prédiction des dégradations")
    print("  ✓ FEC adaptatif (0-75% redondance)")
    print("  ✓ Entrelacement adaptatif (profondeur 2-16)")
    print("  ✓ Score de qualité global")
    print("  ✓ Adaptation anticipative")
    print()
    
    # Crée le scénario complexe
    phases = create_complex_scenario()
    
    # Configuration initiale
    config = RadioConfig(
        use_dsss=True,
        encryption_enabled=True,
        tx_power_dbm=15.0,  # Commence avec puissance modérée
        data_rate=10000,
        mobile_scenario=phases[0]['scenario'],
        environment=phases[0]['environment'],
        enable_rayleigh_fading=True
    )
    
    # Crée le simulateur et contrôleur
    sim = RadioSimulator(config)
    controller = AdaptiveController(
        mode=AdaptiveMode.FULL_AUTO,
        prediction_enabled=True,
        history_size=50
    )
    
    # Stockage des résultats
    results = {
        'time': [],
        'snr': [],
        'ber': [],
        'per': [],
        'quality_score': [],
        'tx_power': [],
        'data_rate': [],
        'spreading_factor': [],
        'fec_enabled': [],
        'fec_redundancy': [],
        'interleaving_enabled': [],
        'interleaving_depth': [],
        'doppler_shift': [],
        'degradation_prob': [],
        'phase_name': [],
        'modulation': []
    }
    
    total_time = 0
    packet_size = 128
    num_packets = 20
    
    print("Début de la simulation...\n")
    
    for phase_idx, phase in enumerate(phases):
        print(f"Phase {phase_idx + 1}/{len(phases)}: {phase['name']}")
        print(f"  Environnement: {phase['environment'].value}")
        print(f"  Vitesse: {phase['scenario'].velocity_ms * 3.6:.1f} km/h")
        print()
        
        # Met à jour la configuration
        sim.config.environment = phase['environment']
        sim.config.mobile_scenario = phase['scenario']
        sim.propagation.environment = phase['environment']
        
        # Simule pendant la durée de la phase
        time_steps = np.linspace(0, phase['duration'], 20)
        
        for t in time_steps:
            # Distance variable selon le scénario
            angle_rad = np.deg2rad(phase['scenario'].angle_deg)
            vx = phase['scenario'].velocity_ms * np.cos(angle_rad)
            vy = phase['scenario'].velocity_ms * np.sin(angle_rad)
            
            x0 = phase['scenario'].distance_m * np.cos(angle_rad)
            y0 = phase['scenario'].distance_m * np.sin(angle_rad)
            
            x = x0 + vx * t
            y = y0 + vy * t
            distance = np.sqrt(x**2 + y**2)
            
            # Simule la transmission
            result = sim.simulate_transmission(
                packet_size_bits=packet_size,
                distance_m=distance,
                num_packets=num_packets,
                time_offset_s=total_time + t
            )
            
            # Génère des données pour estimation
            data = np.random.randint(0, 2, size=packet_size)
            signal_tx, metadata = sim.transmit_packet(data)
            signal_rx, rx_power = sim.channel_propagation(signal_tx, distance, total_time + t)
            received = sim.receive_packet(signal_rx, metadata)
            
            # Estime le canal
            channel = controller.estimate_channel(
                signal_rx=(signal_rx > 0).astype(int)[:len(data)],
                signal_tx=data,
                rx_power_dbm=result.rx_power_dbm,
                noise_floor_dbm=phase['base_noise'],
                timestamp=total_time + t
            )
            
            # Adapte les paramètres
            params = controller.update(channel)
            
            # Applique les nouveaux paramètres
            sim.config.tx_power_dbm = params.tx_power_dbm
            sim.config.data_rate = params.data_rate
            sim.config.use_dsss = params.use_dsss
            
            # Obtient les prédictions
            prediction = controller.predict_degradation()
            
            # Stocke les résultats
            results['time'].append(total_time + t)
            results['snr'].append(result.snr_db)
            results['ber'].append(max(result.ber, 1e-8))  # Évite log(0)
            results['per'].append(result.per)
            results['quality_score'].append(channel.quality_score())
            results['tx_power'].append(params.tx_power_dbm)
            results['data_rate'].append(params.data_rate / 1000)  # En kbps
            results['spreading_factor'].append(params.spreading_factor)
            results['fec_enabled'].append(1 if params.fec_enabled else 0)
            results['fec_redundancy'].append(params.fec_redundancy * 100)  # En %
            results['interleaving_enabled'].append(1 if params.interleaving_enabled else 0)
            results['interleaving_depth'].append(params.interleaving_depth)
            results['doppler_shift'].append(channel.doppler_shift_hz)
            results['degradation_prob'].append(prediction['degradation_probability'] * 100)
            results['phase_name'].append(phase['name'])
            results['modulation'].append('DSSS' if params.use_dsss else 'FHSS')
        
        total_time += phase['duration']
        print(f"  ✓ Phase complétée (temps total: {total_time:.1f}s)\n")
    
    # Statistiques finales
    print("=" * 70)
    print("📊 STATISTIQUES FINALES")
    print("=" * 70)
    
    stats = controller.get_statistics()
    print(f"\nAdaptations:")
    print(f"  Total: {stats['num_adaptations']}")
    print(f"  Changements de puissance: {stats['power_changes']}")
    print(f"  Changements de débit: {stats['rate_changes']}")
    print(f"  Changements de modulation: {stats['modulation_changes']}")
    
    print(f"\nPerformances moyennes:")
    print(f"  SNR: {stats['avg_snr_db']:.1f} dB")
    print(f"  BER: {stats['avg_ber']:.2e}")
    print(f"  Score qualité: {stats['avg_quality_score']:.1f}/100")
    print(f"  Doppler: {stats['avg_doppler_hz']:.1f} Hz")
    
    print(f"\nUtilisation des fonctionnalités avancées:")
    print(f"  FEC activé: {stats['fec_usage_percent']:.1f}% du temps")
    print(f"  Redondance FEC moyenne: {stats['avg_fec_redundancy']:.1f}%")
    print(f"  Entrelacement activé: {stats['interleaving_usage_percent']:.1f}% du temps")
    print(f"  Profondeur entrelacement moyenne: {stats['avg_interleaving_depth']:.1f}")
    
    if 'current_degradation_probability' in stats:
        print(f"\nPrédiction actuelle:")
        print(f"  Probabilité dégradation: {stats['current_degradation_probability']*100:.1f}%")
        print(f"  Tendance qualité: {stats['quality_trend']:+.2f}")
    
    # Génère les graphiques
    plot_intelligent_results(results, phases)
    
    print("\n✓ Graphiques sauvegardés: intelligent_adaptive_*.png")


def plot_intelligent_results(results, phases):
    """Génère les graphiques des résultats"""
    
    fig = plt.figure(figsize=(18, 12))
    
    # Convertit en arrays numpy
    time = np.array(results['time'])
    
    # 1. Score de qualité global
    ax1 = plt.subplot(3, 3, 1)
    ax1.plot(time, results['quality_score'], 'b-', linewidth=2, label='Score qualité')
    ax1.axhline(y=60, color='orange', linestyle='--', alpha=0.5, label='Seuil acceptable')
    ax1.fill_between(time, 0, results['quality_score'], alpha=0.3)
    ax1.set_ylabel('Score qualité (0-100)')
    ax1.set_title('Score de Qualité Global')
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='best')
    add_phase_backgrounds(ax1, phases, time)
    
    # 2. SNR
    ax2 = plt.subplot(3, 3, 2)
    ax2.plot(time, results['snr'], 'g-', linewidth=2)
    ax2.set_ylabel('SNR (dB)')
    ax2.set_title('Rapport Signal/Bruit')
    ax2.grid(True, alpha=0.3)
    add_phase_backgrounds(ax2, phases, time)
    
    # 3. BER
    ax3 = plt.subplot(3, 3, 3)
    ax3.semilogy(time, results['ber'], 'r-', linewidth=2)
    ax3.axhline(y=1e-3, color='orange', linestyle='--', alpha=0.5, label='Seuil bon')
    ax3.set_ylabel('BER')
    ax3.set_title('Taux d\'Erreur Binaire')
    ax3.grid(True, alpha=0.3)
    ax3.legend(loc='best')
    add_phase_backgrounds(ax3, phases, time)
    
    # 4. Puissance TX adaptative
    ax4 = plt.subplot(3, 3, 4)
    ax4.plot(time, results['tx_power'], 'm-', linewidth=2, marker='o', markersize=3)
    ax4.set_ylabel('Puissance TX (dBm)')
    ax4.set_title('Puissance d\'Émission Adaptative')
    ax4.grid(True, alpha=0.3)
    add_phase_backgrounds(ax4, phases, time)
    
    # 5. FEC adaptatif
    ax5 = plt.subplot(3, 3, 5)
    ax5.fill_between(time, 0, results['fec_redundancy'], 
                     where=np.array(results['fec_enabled']) > 0,
                     color='cyan', alpha=0.6, label='FEC actif')
    ax5.plot(time, results['fec_redundancy'], 'c-', linewidth=2)
    ax5.set_ylabel('Redondance FEC (%)')
    ax5.set_title('Codage Correcteur d\'Erreurs Adaptatif')
    ax5.set_ylim([0, 80])
    ax5.grid(True, alpha=0.3)
    ax5.legend(loc='best')
    add_phase_backgrounds(ax5, phases, time)
    
    # 6. Entrelacement adaptatif
    ax6 = plt.subplot(3, 3, 6)
    ax6.fill_between(time, 0, results['interleaving_depth'],
                     where=np.array(results['interleaving_enabled']) > 0,
                     color='yellow', alpha=0.6, label='Entrelacement actif')
    ax6.plot(time, results['interleaving_depth'], 'y-', linewidth=2)
    ax6.set_ylabel('Profondeur entrelacement')
    ax6.set_title('Entrelacement Adaptatif')
    ax6.grid(True, alpha=0.3)
    ax6.legend(loc='best')
    add_phase_backgrounds(ax6, phases, time)
    
    # 7. Débit adaptatif
    ax7 = plt.subplot(3, 3, 7)
    ax7.step(time, results['data_rate'], 'b-', linewidth=2, where='post')
    ax7.set_ylabel('Débit (kbps)')
    ax7.set_xlabel('Temps (s)')
    ax7.set_title('Débit de Données Adaptatif')
    ax7.grid(True, alpha=0.3)
    add_phase_backgrounds(ax7, phases, time)
    
    # 8. Probabilité de dégradation prédite
    ax8 = plt.subplot(3, 3, 8)
    ax8.plot(time, results['degradation_prob'], 'r-', linewidth=2)
    ax8.fill_between(time, 0, results['degradation_prob'], 
                     where=np.array(results['degradation_prob']) > 60,
                     color='red', alpha=0.3, label='Alerte')
    ax8.axhline(y=60, color='red', linestyle='--', alpha=0.5)
    ax8.set_ylabel('Probabilité (%)')
    ax8.set_xlabel('Temps (s)')
    ax8.set_title('Prédiction de Dégradation')
    ax8.set_ylim([0, 100])
    ax8.grid(True, alpha=0.3)
    ax8.legend(loc='best')
    add_phase_backgrounds(ax8, phases, time)
    
    # 9. Spreading Factor
    ax9 = plt.subplot(3, 3, 9)
    ax9.step(time, results['spreading_factor'], 'g-', linewidth=2, where='post')
    ax9.set_ylabel('Spreading Factor')
    ax9.set_xlabel('Temps (s)')
    ax9.set_title('Facteur d\'Étalement Adaptatif')
    ax9.grid(True, alpha=0.3)
    add_phase_backgrounds(ax9, phases, time)
    
    plt.suptitle('Système Adaptatif Intelligent - Simulation Complète', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('intelligent_adaptive_complete.png', dpi=200, bbox_inches='tight')
    print("  → intelligent_adaptive_complete.png")
    
    # Graphique supplémentaire: Résumé par phase
    plot_phase_summary(results, phases)
    
    plt.show()


def add_phase_backgrounds(ax, phases, time):
    """Ajoute des arrière-plans colorés pour chaque phase"""
    colors = ['lightblue', 'lightgreen', 'lightyellow', 'lightcoral', 'lightgray']
    current_time = 0
    
    for i, phase in enumerate(phases):
        ax.axvspan(current_time, current_time + phase['duration'], 
                  alpha=0.1, color=colors[i % len(colors)])
        current_time += phase['duration']


def plot_phase_summary(results, phases):
    """Graphique résumé par phase"""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    phase_names = []
    avg_quality = []
    avg_snr = []
    avg_ber = []
    fec_usage = []
    
    current_time = 0
    for phase in phases:
        phase_names.append(phase['name'])
        
        # Filtre les données de cette phase
        mask = (np.array(results['time']) >= current_time) & \
               (np.array(results['time']) < current_time + phase['duration'])
        
        avg_quality.append(np.mean(np.array(results['quality_score'])[mask]))
        avg_snr.append(np.mean(np.array(results['snr'])[mask]))
        avg_ber.append(np.mean(np.array(results['ber'])[mask]))
        fec_usage.append(np.mean(np.array(results['fec_enabled'])[mask]) * 100)
        
        current_time += phase['duration']
    
    # Graphique 1: Score de qualité par phase
    axes[0, 0].bar(range(len(phase_names)), avg_quality, color='skyblue', edgecolor='navy')
    axes[0, 0].axhline(y=60, color='orange', linestyle='--', label='Seuil acceptable')
    axes[0, 0].set_ylabel('Score qualité moyen')
    axes[0, 0].set_title('Qualité par Phase')
    axes[0, 0].set_xticks(range(len(phase_names)))
    axes[0, 0].set_xticklabels([f'P{i+1}' for i in range(len(phase_names))], rotation=0)
    axes[0, 0].grid(True, alpha=0.3, axis='y')
    axes[0, 0].legend()
    
    # Graphique 2: SNR moyen par phase
    axes[0, 1].bar(range(len(phase_names)), avg_snr, color='lightgreen', edgecolor='darkgreen')
    axes[0, 1].set_ylabel('SNR moyen (dB)')
    axes[0, 1].set_title('SNR par Phase')
    axes[0, 1].set_xticks(range(len(phase_names)))
    axes[0, 1].set_xticklabels([f'P{i+1}' for i in range(len(phase_names))], rotation=0)
    axes[0, 1].grid(True, alpha=0.3, axis='y')
    
    # Graphique 3: BER moyen par phase (log scale)
    axes[1, 0].bar(range(len(phase_names)), avg_ber, color='lightcoral', edgecolor='darkred')
    axes[1, 0].set_ylabel('BER moyen')
    axes[1, 0].set_title('BER par Phase')
    axes[1, 0].set_yscale('log')
    axes[1, 0].set_xticks(range(len(phase_names)))
    axes[1, 0].set_xticklabels([f'P{i+1}' for i in range(len(phase_names))], rotation=0)
    axes[1, 0].grid(True, alpha=0.3, axis='y')
    
    # Graphique 4: Usage FEC par phase
    axes[1, 1].bar(range(len(phase_names)), fec_usage, color='plum', edgecolor='purple')
    axes[1, 1].set_ylabel('Utilisation FEC (%)')
    axes[1, 1].set_title('Usage FEC par Phase')
    axes[1, 1].set_xticks(range(len(phase_names)))
    axes[1, 1].set_xticklabels([f'P{i+1}' for i in range(len(phase_names))], rotation=0)
    axes[1, 1].set_ylim([0, 100])
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    
    # Légende des phases
    legend_text = '\n'.join([f'P{i+1}: {name}' for i, name in enumerate(phase_names)])
    fig.text(0.02, 0.98, legend_text, fontsize=9, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.suptitle('Analyse par Phase - Système Adaptatif', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('intelligent_adaptive_phases.png', dpi=200, bbox_inches='tight')
    print("  → intelligent_adaptive_phases.png")


if __name__ == "__main__":
    simulate_intelligent_system()
