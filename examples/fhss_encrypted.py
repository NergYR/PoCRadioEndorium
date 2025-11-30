"""
Script d'exemple : FHSS chiffré (AES-CTR + HMAC)
"""

from src.simulator import RadioSimulator, RadioConfig
from src.propagation import Environment

print("=" * 60)
print("FHSS chiffré - Scénario ext. ouvert")
print("=" * 60)

config = RadioConfig(
    use_dsss=False,
    num_channels=79,
    hop_duration=0.05,
    data_rate=10000,
    frequency_mhz=915.0,
    tx_power_dbm=20.0,
    environment=Environment.OUTDOOR_OPEN,
    encryption_enabled=True,
    password='airsoft2024'
)

sim = RadioSimulator(config)

for d in [100, 300, 500, 800]:
    res = sim.simulate_transmission(packet_size_bits=256, distance_m=d, num_packets=100)
    print(f"{d:>4} m | BER={res.ber:.2e} | PER={res.per:.2%} | SNR={res.snr_db:.1f} dB | RX={res.rx_power_dbm:.1f} dBm")
