"""
Script d'exemple : Génération de graphiques d'analyse de portée (chiffré)
"""

from src.simulator import RadioSimulator, RadioConfig
from src.propagation import Environment

print("=" * 60)
print("Génération de l'analyse de portée (AES-CTR + HMAC)")
print("=" * 60)

# Configuration pour terrain d'airsoft extérieur
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

print("\nConfiguration:")
print(f"  Modulation: DSSS")
print(f"  Facteur d'étalement: {config.chip_rate // config.data_rate}")
print(f"  Débit: {config.data_rate / 1000} kbps")
print(f"  Fréquence: {config.frequency_mhz} MHz")
print(f"  Puissance TX: {config.tx_power_dbm} dBm")
print(f"  Environnement: {config.environment.value}")

sim = RadioSimulator(config)

print("\nGénération des graphiques...")
print("(Cela peut prendre quelques minutes)")

# Génère l'analyse complète
sim.plot_range_analysis(max_distance_m=2000, step_m=50)

print("\n✓ Graphiques générés avec succès!")
print("  Fichier: range_analysis.png")
