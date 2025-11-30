"""
Test sans chiffrement pour voir les performances pures
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from modulation import DSSSModulator, add_awgn
from propagation import PropagationModel, Environment
import numpy as np

print("=" * 70)
print("TEST PERFORMANCES - Sans chiffrement")
print("=" * 70)

# Configuration DSSS
dsss = DSSSModulator(chip_rate=1000000, data_rate=10000)
prop = PropagationModel(frequency_mhz=915.0, tx_power_dbm=20.0)

print(f"\nConfiguration DSSS:")
print(f"  Facteur d'étalement: {dsss.spreading_factor}")
print(f"  Gain de traitement: {dsss.get_processing_gain_db():.1f} dB")
print(f"  Débit: 10 kbps")
print(f"  Environnement: Extérieur ouvert")

test_distances = [100, 300, 500, 1000, 1500]

print(f"\nSimulation de transmission:\n")

for distance in test_distances:
    # Génère des données aléatoires
    num_packets = 100
    packet_size = 128
    total_errors = 0
    packet_errors = 0
    
    for _ in range(num_packets):
        data = np.random.randint(0, 2, size=packet_size)
        
        # Modulation
        signal = dsss.spread(data)
        
        # Canal avec bruit
        rx_power = prop.received_power(distance, Environment.OUTDOOR_OPEN)
        noise_floor = -174 + 10 * np.log10(10000)  # 10 kbps
        snr_db = rx_power - noise_floor
        noisy_signal = add_awgn(signal, snr_db)
        
        # Démodulation
        recovered = dsss.despread(noisy_signal)
        
        # Compte les erreurs
        errors = np.sum(data != recovered)
        total_errors += errors
        if errors > 0:
            packet_errors += 1
    
    ber = total_errors / (num_packets * packet_size)
    per = packet_errors / num_packets
    
    link_ok, margin = prop.link_budget(distance, Environment.OUTDOOR_OPEN)
    
    print(f"Distance: {distance}m")
    print(f"  ├─ BER: {ber:.2e}")
    print(f"  ├─ PER: {per:.2%}")
    print(f"  ├─ SNR: {snr_db:.1f} dB")
    print(f"  ├─ Puissance reçue: {rx_power:.1f} dBm")
    print(f"  ├─ Marge: {margin:.1f} dB")
    
    if ber < 1e-5:
        quality = "Excellente ✓✓✓"
    elif ber < 1e-3:
        quality = "Très bonne ✓✓"
    elif ber < 1e-2:
        quality = "Bonne ✓"
    elif ber < 5e-2:
        quality = "Acceptable ⚠"
    else:
        quality = "Médiocre ✗"
    
    print(f"  └─ Qualité: {quality}\n")

print("=" * 70)
print("Note: Sans chiffrement, les performances sont optimales.")
print("Le chiffrement ajoute de l'overhead mais assure la confidentialité.")
print("=" * 70)
