"""
Script d'exemple : Comparaison DSSS vs FHSS (chiffré)
"""

from src.simulator import RadioSimulator, RadioConfig
from src.propagation import Environment

print("=" * 60)
print("Comparaison DSSS vs FHSS pour Airsoft (AES-CTR + HMAC)")
print("=" * 60)

# Configuration commune
base_config = {
    'data_rate': 10000,
    'frequency_mhz': 915.0,
    'tx_power_dbm': 20.0,
    'encryption_enabled': True,
    'password': 'airsoft2024',
    'environment': Environment.OUTDOOR_OPEN
}

# Test DSSS
print("\n--- Configuration DSSS ---")
config_dsss = RadioConfig(**base_config, use_dsss=True)
sim_dsss = RadioSimulator(config_dsss)

result_dsss = sim_dsss.simulate_transmission(packet_size_bits=256, distance_m=500, num_packets=100)
print(f"BER: {result_dsss.ber:.2e}")
print(f"PER: {result_dsss.per:.2%}")
print(f"SNR: {result_dsss.snr_db:.1f} dB")
print(f"Débit effectif: {result_dsss.throughput_kbps:.2f} kbps")

# Test FHSS
print("\n--- Configuration FHSS ---")
config_fhss = RadioConfig(**base_config, use_dsss=False)
sim_fhss = RadioSimulator(config_fhss)

result_fhss = sim_fhss.simulate_transmission(packet_size_bits=256, distance_m=500, num_packets=100)
print(f"BER: {result_fhss.ber:.2e}")
print(f"PER: {result_fhss.per:.2%}")
print(f"SNR: {result_fhss.snr_db:.1f} dB")
print(f"Débit effectif: {result_fhss.throughput_kbps:.2f} kbps")

print("\n" + "=" * 60)
print("Conclusion:")
if result_dsss.ber < result_fhss.ber:
    print("✓ DSSS offre de meilleures performances pour ce scénario")
else:
    print("✓ FHSS offre de meilleures performances pour ce scénario")
print("=" * 60)
