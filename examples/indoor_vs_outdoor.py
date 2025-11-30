"""
Script d'exemple : Comparaison environnements intérieur/extérieur (chiffré)
"""

from src.simulator import RadioSimulator, RadioConfig
from src.propagation import Environment

print("=" * 60)
print("Analyse de portée : Intérieur vs Extérieur (AES-CTR + HMAC)")
print("=" * 60)

environments = [
    (Environment.OUTDOOR_OPEN, "Extérieur terrain ouvert (forêt dégagée)"),
    (Environment.OUTDOOR_URBAN, "Extérieur urbain (ville)"),
    (Environment.INDOOR_BUILDING, "Intérieur bâtiment (CQB indoor)")
]

distance_test = 200  # mètres

for env, description in environments:
    print(f"\n--- {description} ---")
    
    config = RadioConfig(
        use_dsss=True,
        environment=env,
        encryption_enabled=True,
        password='airsoft2024'
    )
    
    sim = RadioSimulator(config)
    result = sim.simulate_transmission(
        packet_size_bits=256, 
        distance_m=distance_test, 
        num_packets=100
    )
    
    print(f"Distance de test: {distance_test}m")
    print(f"  Puissance reçue: {result.rx_power_dbm:.1f} dBm")
    print(f"  Marge de liaison: {result.link_margin_db:.1f} dB")
    print(f"  BER: {result.ber:.2e}")
    print(f"  PER: {result.per:.2%}")
    
    if result.link_margin_db > 10:
        print(f"  ✓ Liaison excellente")
    elif result.link_margin_db > 0:
        print(f"  ⚠ Liaison acceptable")
    else:
        print(f"  ✗ Liaison impossible")

print("\n" + "=" * 60)
