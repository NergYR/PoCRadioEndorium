"""
Exécution rapide d'une simulation chiffrée (AES-CTR + HMAC)
"""
from src.simulator import RadioSimulator, RadioConfig
from src.propagation import Environment

config = RadioConfig(
    use_dsss=True,
    data_rate=10000,
    environment=Environment.OUTDOOR_OPEN,
    encryption_enabled=True,
    password="airsoft2024"
)

sim = RadioSimulator(config)

for d in [100, 300, 500, 1000]:
    res = sim.simulate_transmission(packet_size_bits=256, distance_m=d, num_packets=50)
    print(f"{d:>4} m | BER={res.ber:.2e} | PER={res.per:.2%} | SNR={res.snr_db:.1f} dB | RX={res.rx_power_dbm:.1f} dBm")
