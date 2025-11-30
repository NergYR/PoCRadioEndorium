"""
Export de signaux pour GNURadio
Format: fichiers binaires .bin (complex float32) et .csv
"""

import numpy as np
import struct
from src.simulator import RadioSimulator, RadioConfig
from src.propagation import Environment
from src.modulation import DSSSModulator, add_awgn

print("=" * 70)
print("EXPORT POUR GNURADIO")
print("=" * 70)

# Configuration
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

# ========== Export 1: Signal DSSS brut ==========
print("\n[1/3] Export signal DSSS...")

# Génère des données
data = np.random.randint(0, 2, size=1000, dtype=np.uint8)
signal, metadata = sim.transmit_packet(data)

# Normalise et convertit en complexe (I+jQ)
# On simule une modulation BPSK: signal réel → parties I, Q=0
signal_normalized = signal / np.max(np.abs(signal))
signal_complex = signal_normalized.astype(np.float32) + 0j

# Export binaire (format GNURadio: complex64 = float32 I, float32 Q)
with open('gnuradio_dsss_signal.bin', 'wb') as f:
    for sample in signal_complex:
        f.write(struct.pack('ff', sample.real, sample.imag))

print(f"   ✓ Signal exporté: gnuradio_dsss_signal.bin")
print(f"     Échantillons: {len(signal_complex)}")
print(f"     Taux: {config.chip_rate} Hz")
print(f"     Format: complex float32 (IQ)")

# ========== Export 2: Signal avec canal simulé ==========
print("\n[2/3] Export signal avec canal (SNR=15dB)...")

# Ajoute du bruit réaliste
noisy_signal = add_awgn(signal, snr_db=15)
noisy_normalized = noisy_signal / np.max(np.abs(noisy_signal))
noisy_complex = noisy_normalized.astype(np.float32) + 0j

with open('gnuradio_dsss_noisy.bin', 'wb') as f:
    for sample in noisy_complex:
        f.write(struct.pack('ff', sample.real, sample.imag))

print(f"   ✓ Signal bruité exporté: gnuradio_dsss_noisy.bin")
print(f"     SNR: 15 dB")
print(f"     Échantillons: {len(noisy_complex)}")

# ========== Export 3: Métadonnées CSV ==========
print("\n[3/3] Export métadonnées...")

metadata_info = f"""# GNURadio Signal Metadata
# Générateur: PoCRadio - Simulation Airsoft
sample_rate,{config.chip_rate}
center_frequency,{config.frequency_mhz * 1e6}
modulation,DSSS-BPSK
spreading_factor,{config.chip_rate // config.data_rate}
data_rate,{config.data_rate}
encryption,AES-256-CTR-HMAC
tx_power_dbm,{config.tx_power_dbm}
samples,{len(signal_complex)}
format,complex_float32
"""

with open('gnuradio_metadata.csv', 'w') as f:
    f.write(metadata_info)

print(f"   ✓ Métadonnées: gnuradio_metadata.csv")

# ========== Export 4: Séquence PN pour désétalement ==========
print("\n[4/4] Export séquence PN...")

modulator = DSSSModulator(config.chip_rate, config.data_rate)
pn_sequence = modulator.pn_sequence

# Export en float32 pour GNURadio
pn_float = pn_sequence.astype(np.float32)
pn_float.tofile('gnuradio_pn_sequence.bin')

print(f"   ✓ Séquence PN: gnuradio_pn_sequence.bin")
print(f"     Longueur: {len(pn_sequence)} chips")
print(f"     Valeurs: +1/-1")

print("\n" + "=" * 70)
print("✓ EXPORT TERMINÉ")
print("=" * 70)
print("\nFichiers GNURadio générés:")
print("  • gnuradio_dsss_signal.bin    - Signal DSSS propre")
print("  • gnuradio_dsss_noisy.bin     - Signal avec bruit (SNR=15dB)")
print("  • gnuradio_pn_sequence.bin    - Séquence PN pour désétalement")
print("  • gnuradio_metadata.csv       - Paramètres du signal")
print("\nUtilisation dans GNURadio:")
print("  1. File Source → Type: Complex, Repeat: No")
print("  2. Sample Rate: 1000000 (1 MHz)")
print("  3. Pour désétaler: multiplier par PN sequence répétée")
print("  4. Intégrer sur 100 échantillons (spreading factor)")
print("=" * 70)
