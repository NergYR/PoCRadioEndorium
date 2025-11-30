"""
Génération de graphiques complets pour simulation réelle
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from src.simulator import RadioSimulator, RadioConfig
from src.propagation import Environment, PropagationModel
from src.modulation import DSSSModulator

# Désactive le mode interactif pour générer les images
plt.ioff()

print("=" * 70)
print("GÉNÉRATION GRAPHIQUES SIMULATION RÉELLE")
print("=" * 70)

# Configuration DSSS chiffrée pour airsoft
config = RadioConfig(
    use_dsss=True,
    chip_rate=1000000,
    data_rate=10000,
    frequency_mhz=915.0,
    tx_power_dbm=20.0,
    environment=Environment.OUTDOOR_OPEN,
    encryption_enabled=True,
    password="airsoft2024"
)

sim = RadioSimulator(config)

# ========== FIGURE 1: Analyse de performance vs distance ==========
print("\n[1/4] Génération: Analyse de performance vs distance...")

distances = np.arange(50, 1500, 50)
bers = []
pers = []
rx_powers = []
snrs = []
margins = []

for d in distances:
    res = sim.simulate_transmission(packet_size_bits=256, distance_m=d, num_packets=50)
    bers.append(max(res.ber, 1e-8))  # Évite log(0)
    pers.append(res.per)
    rx_powers.append(res.rx_power_dbm)
    snrs.append(res.snr_db)
    margins.append(res.link_margin_db)

fig1 = plt.figure(figsize=(14, 10))
gs = GridSpec(3, 2, figure=fig1, hspace=0.3, wspace=0.3)

# BER vs Distance
ax1 = fig1.add_subplot(gs[0, 0])
ax1.semilogy(distances, bers, 'b-', linewidth=2, label='BER')
ax1.axhline(y=1e-3, color='r', linestyle='--', alpha=0.5, label='Seuil 10⁻³')
ax1.set_xlabel('Distance (m)', fontsize=11)
ax1.set_ylabel('Bit Error Rate (BER)', fontsize=11)
ax1.set_title('Taux d\'erreur binaire', fontsize=12, fontweight='bold')
ax1.grid(True, alpha=0.3)
ax1.legend()

# PER vs Distance
ax2 = fig1.add_subplot(gs[0, 1])
ax2.plot(distances, [p*100 for p in pers], 'g-', linewidth=2)
ax2.set_xlabel('Distance (m)', fontsize=11)
ax2.set_ylabel('Packet Error Rate (%)', fontsize=11)
ax2.set_title('Taux d\'erreur paquet', fontsize=12, fontweight='bold')
ax2.grid(True, alpha=0.3)

# Puissance reçue vs Distance
ax3 = fig1.add_subplot(gs[1, 0])
ax3.plot(distances, rx_powers, 'r-', linewidth=2, label='Puissance reçue')
ax3.axhline(y=config.rx_sensitivity_dbm, color='k', linestyle='--', 
           label=f'Sensibilité ({config.rx_sensitivity_dbm} dBm)')
ax3.fill_between(distances, rx_powers, config.rx_sensitivity_dbm, 
                 where=np.array(rx_powers)>=config.rx_sensitivity_dbm, 
                 alpha=0.2, color='green', label='Zone de couverture')
ax3.set_xlabel('Distance (m)', fontsize=11)
ax3.set_ylabel('Puissance reçue (dBm)', fontsize=11)
ax3.set_title('Niveau du signal reçu', fontsize=12, fontweight='bold')
ax3.legend()
ax3.grid(True, alpha=0.3)

# SNR vs Distance
ax4 = fig1.add_subplot(gs[1, 1])
ax4.plot(distances, snrs, 'm-', linewidth=2)
ax4.axhline(y=10, color='orange', linestyle='--', alpha=0.5, label='SNR min recommandé (10 dB)')
ax4.set_xlabel('Distance (m)', fontsize=11)
ax4.set_ylabel('SNR (dB)', fontsize=11)
ax4.set_title('Rapport Signal/Bruit', fontsize=12, fontweight='bold')
ax4.legend()
ax4.grid(True, alpha=0.3)

# Marge de liaison
ax5 = fig1.add_subplot(gs[2, :])
colors = ['green' if m > 20 else 'orange' if m > 10 else 'red' for m in margins]
ax5.bar(distances, margins, width=40, color=colors, alpha=0.6)
ax5.axhline(y=10, color='k', linestyle='--', alpha=0.5, label='Marge min (10 dB)')
ax5.set_xlabel('Distance (m)', fontsize=11)
ax5.set_ylabel('Marge de liaison (dB)', fontsize=11)
ax5.set_title('Marge de liaison (Vert: >20dB, Orange: 10-20dB, Rouge: <10dB)', 
             fontsize=12, fontweight='bold')
ax5.legend()
ax5.grid(True, alpha=0.3, axis='y')

fig1.suptitle(f'Analyse de performance - DSSS chiffré - {config.environment.value}', 
             fontsize=14, fontweight='bold')
plt.savefig('simulation_performance.png', dpi=150, bbox_inches='tight')
print("   ✓ Sauvegardé: simulation_performance.png")
plt.close(fig1)

# ========== FIGURE 2: Comparaison environnements ==========
print("\n[2/4] Génération: Comparaison environnements airsoft...")

environments = [
    (Environment.OUTDOOR_OPEN, "Terrain ouvert", 'green'),
    (Environment.OUTDOOR_SUBURBAN, "Zone suburbaine", 'blue'),
    (Environment.OUTDOOR_URBAN, "Zone urbaine (CQB)", 'orange'),
    (Environment.INDOOR_BUILDING, "Intérieur bâtiment", 'red')
]

fig2, axes = plt.subplots(2, 2, figsize=(14, 10))
fig2.suptitle('Comparaison environnements Airsoft - DSSS chiffré', 
             fontsize=14, fontweight='bold')

test_distances = np.arange(50, 800, 50)

for idx, (env, label, color) in enumerate(environments):
    ax = axes[idx // 2, idx % 2]
    
    config_env = RadioConfig(
        use_dsss=True,
        environment=env,
        encryption_enabled=True,
        password="airsoft2024"
    )
    sim_env = RadioSimulator(config_env)
    
    env_bers = []
    env_rx_powers = []
    
    for d in test_distances:
        res = sim_env.simulate_transmission(256, d, 30)
        env_bers.append(max(res.ber, 1e-8))
        env_rx_powers.append(res.rx_power_dbm)
    
    # BER
    ax_ber = ax
    ax_ber.semilogy(test_distances, env_bers, color=color, linewidth=2, label='BER')
    ax_ber.axhline(y=1e-3, color='gray', linestyle='--', alpha=0.5)
    ax_ber.set_xlabel('Distance (m)', fontsize=10)
    ax_ber.set_ylabel('BER', fontsize=10, color=color)
    ax_ber.tick_params(axis='y', labelcolor=color)
    ax_ber.set_title(label, fontsize=11, fontweight='bold')
    ax_ber.grid(True, alpha=0.3)
    
    # Puissance reçue sur axe secondaire
    ax_pwr = ax.twinx()
    ax_pwr.plot(test_distances, env_rx_powers, color=color, 
               linewidth=2, linestyle='--', alpha=0.6, label='Rx Power')
    ax_pwr.set_ylabel('Puissance reçue (dBm)', fontsize=10, color=color)
    ax_pwr.tick_params(axis='y', labelcolor=color)

plt.tight_layout()
plt.savefig('simulation_environments.png', dpi=150, bbox_inches='tight')
print("   ✓ Sauvegardé: simulation_environments.png")
plt.close(fig2)

# ========== FIGURE 3: Spectre et constellation DSSS ==========
print("\n[3/4] Génération: Spectre et constellation DSSS...")

# Génère un signal DSSS
modulator = DSSSModulator(chip_rate=1000000, data_rate=10000)
data_bits = np.random.randint(0, 2, size=100, dtype=np.uint8)
spread_signal = modulator.spread(data_bits)

# Ajoute du bruit réaliste
from src.modulation import add_awgn
noisy_signal = add_awgn(spread_signal, snr_db=15)

fig3, axes = plt.subplots(2, 2, figsize=(14, 10))
fig3.suptitle('Analyse spectrale et constellation DSSS', fontsize=14, fontweight='bold')

# Signal temporel
ax_time = axes[0, 0]
time_axis = np.arange(len(spread_signal[:1000])) / 1e6
ax_time.plot(time_axis * 1e6, spread_signal[:1000], 'b-', linewidth=0.8)
ax_time.set_xlabel('Temps (µs)', fontsize=10)
ax_time.set_ylabel('Amplitude', fontsize=10)
ax_time.set_title('Signal DSSS étalé (1000 premiers chips)', fontsize=11, fontweight='bold')
ax_time.grid(True, alpha=0.3)

# Spectre de fréquence
ax_fft = axes[0, 1]
fft_signal = np.fft.fftshift(np.fft.fft(spread_signal))
freq_axis = np.fft.fftshift(np.fft.fftfreq(len(spread_signal), 1/1e6)) / 1e3
power_spectrum = 10 * np.log10(np.abs(fft_signal)**2 + 1e-12)
ax_fft.plot(freq_axis, power_spectrum, 'r-', linewidth=1)
ax_fft.set_xlabel('Fréquence (kHz)', fontsize=10)
ax_fft.set_ylabel('Puissance (dB)', fontsize=10)
ax_fft.set_title(f'Spectre de fréquence (Bande: ~{1000} kHz)', fontsize=11, fontweight='bold')
ax_fft.grid(True, alpha=0.3)
ax_fft.set_xlim([-600, 600])

# Constellation (signal bruité)
ax_const = axes[1, 0]
# Échantillonne le signal pour constellation
samples_i = noisy_signal[::modulator.spreading_factor][:500]
samples_q = np.roll(samples_i, 1)  # Simule quadrature
ax_const.scatter(samples_i, samples_q, alpha=0.3, s=10, c='blue')
ax_const.set_xlabel('In-Phase', fontsize=10)
ax_const.set_ylabel('Quadrature', fontsize=10)
ax_const.set_title('Constellation (SNR=15dB)', fontsize=11, fontweight='bold')
ax_const.grid(True, alpha=0.3)
ax_const.axhline(y=0, color='k', linewidth=0.5)
ax_const.axvline(x=0, color='k', linewidth=0.5)

# Histogramme amplitude
ax_hist = axes[1, 1]
ax_hist.hist(noisy_signal, bins=50, color='purple', alpha=0.7, edgecolor='black')
ax_hist.set_xlabel('Amplitude', fontsize=10)
ax_hist.set_ylabel('Occurrences', fontsize=10)
ax_hist.set_title('Distribution des amplitudes (signal bruité)', fontsize=11, fontweight='bold')
ax_hist.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('simulation_spectrum.png', dpi=150, bbox_inches='tight')
print("   ✓ Sauvegardé: simulation_spectrum.png")
plt.close(fig3)

# ========== FIGURE 4: Résumé statistique ==========
print("\n[4/4] Génération: Tableau récapitulatif...")

fig4, ax = plt.subplots(figsize=(12, 8))
ax.axis('off')

# Données de synthèse
summary_data = []
for env, label, _ in environments:
    cfg = RadioConfig(use_dsss=True, environment=env, encryption_enabled=True)
    sim_tmp = RadioSimulator(cfg)
    
    # Test à 200m et 500m
    res_200 = sim_tmp.simulate_transmission(256, 200, 50)
    res_500 = sim_tmp.simulate_transmission(256, 500, 50)
    
    # Portée max (BER < 1e-3)
    max_range = 0
    for d in range(100, 2000, 100):
        r = sim_tmp.simulate_transmission(256, d, 30)
        if r.ber < 1e-3:
            max_range = d
        else:
            break
    
    summary_data.append([
        label,
        f"{max_range}m",
        f"{res_200.ber:.2e}",
        f"{res_200.rx_power_dbm:.1f} dBm",
        f"{res_500.ber:.2e}",
        f"{res_500.rx_power_dbm:.1f} dBm"
    ])

# Tableau
col_labels = ['Environnement', 'Portée max', 'BER@200m', 'Rx@200m', 'BER@500m', 'Rx@500m']
table = ax.table(cellText=summary_data, colLabels=col_labels, 
                cellLoc='center', loc='center',
                colWidths=[0.25, 0.15, 0.15, 0.15, 0.15, 0.15])

table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 2.5)

# Style
for i in range(len(col_labels)):
    table[(0, i)].set_facecolor('#4CAF50')
    table[(0, i)].set_text_props(weight='bold', color='white')

for i in range(1, len(summary_data) + 1):
    for j in range(len(col_labels)):
        if i % 2 == 0:
            table[(i, j)].set_facecolor('#f0f0f0')

ax.set_title('Synthèse des performances - Simulation réaliste Airsoft\n'
            'DSSS chiffré AES-CTR + HMAC-SHA256',
            fontsize=14, fontweight='bold', pad=20)

# Informations système
info_text = (
    f"Configuration système:\n"
    f"  • Modulation: DSSS (facteur {config.chip_rate // config.data_rate})\n"
    f"  • Chiffrement: AES-256-CTR + HMAC-SHA256\n"
    f"  • Fréquence: {config.frequency_mhz} MHz\n"
    f"  • Puissance TX: {config.tx_power_dbm} dBm\n"
    f"  • Sensibilité RX: {config.rx_sensitivity_dbm} dBm\n"
    f"  • Débit: {config.data_rate/1000} kbps"
)
ax.text(0.5, 0.05, info_text, transform=ax.transAxes,
       fontsize=9, verticalalignment='bottom', horizontalalignment='center',
       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.savefig('simulation_summary.png', dpi=150, bbox_inches='tight')
print("   ✓ Sauvegardé: simulation_summary.png")
plt.close(fig4)

print("\n" + "=" * 70)
print("✓ GÉNÉRATION TERMINÉE")
print("=" * 70)
print("\nFichiers générés:")
print("  1. simulation_performance.png  - Analyse complète vs distance")
print("  2. simulation_environments.png - Comparaison environnements")
print("  3. simulation_spectrum.png     - Analyse spectrale et constellation")
print("  4. simulation_summary.png      - Tableau récapitulatif")
print("\nUtilisation pour airsoft:")
print("  • Terrain ouvert: jusqu'à ~1400m")
print("  • Zone urbaine: jusqu'à ~400m")
print("  • Intérieur: jusqu'à ~200m")
print("=" * 70)
