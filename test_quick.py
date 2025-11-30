"""
Test rapide du simulateur complet
"""

import sys
import os

# Ajoute le répertoire src au path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Import direct depuis les modules
from modulation import DSSSModulator, FHSSModulator, add_awgn
from crypto import RadioCrypto, MessageAuthenticator
from propagation import PropagationModel, Environment

# Import du simulateur
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Mode non-interactif
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import List, Tuple

# Copie des classes nécessaires
@dataclass
class RadioConfig:
    """Configuration du système radio"""
    chip_rate: int = 1000000
    data_rate: int = 10000
    use_dsss: bool = True
    num_channels: int = 50
    hop_duration: float = 0.1
    frequency_mhz: float = 915.0
    tx_power_dbm: float = 20.0
    rx_sensitivity_dbm: float = -110.0
    tx_gain_dbi: float = 2.0
    rx_gain_dbi: float = 2.0
    encryption_enabled: bool = True
    password: str = "airsoft2024"
    environment: Environment = Environment.OUTDOOR_OPEN

@dataclass
class SimulationResult:
    """Résultats de simulation"""
    ber: float
    per: float
    snr_db: float
    rx_power_dbm: float
    link_margin_db: float
    throughput_kbps: float
    latency_ms: float

class RadioSimulator:
    """Simulateur de système radio complet"""
    
    def __init__(self, config: RadioConfig):
        self.config = config
        if config.use_dsss:
            self.modulator = DSSSModulator(config.chip_rate, config.data_rate)
        else:
            self.modulator = FHSSModulator(config.num_channels, config.hop_duration, 
                                          config.frequency_mhz * 1e6)
        self.crypto = RadioCrypto(config.password) if config.encryption_enabled else None
        self.authenticator = MessageAuthenticator() if config.encryption_enabled else None
        self.propagation = PropagationModel(config.frequency_mhz, config.tx_power_dbm)
    
    def transmit_packet(self, data: np.ndarray) -> Tuple[np.ndarray, dict]:
        metadata = {}
        if self.crypto:
            iv, ciphertext, shape, dtype = self.crypto.encrypt_array(data)
            data_to_transmit = np.unpackbits(np.frombuffer(ciphertext, dtype=np.uint8))
            metadata['iv'] = iv
            metadata['shape'] = shape
            metadata['dtype'] = dtype
            metadata['encrypted'] = True
        else:
            data_to_transmit = data
            metadata['encrypted'] = False
        
        if self.config.use_dsss:
            signal = self.modulator.spread(data_to_transmit)
            metadata['modulation'] = 'DSSS'
            metadata['spreading_factor'] = self.modulator.spreading_factor
        else:
            signal = data_to_transmit
            metadata['modulation'] = 'FHSS'
            metadata['hop_rate'] = self.modulator.get_hop_rate()
        
        metadata['signal_length'] = len(signal)
        return signal, metadata
    
    def channel_propagation(self, signal: np.ndarray, distance_m: float) -> Tuple[np.ndarray, float]:
        rx_power_dbm = self.propagation.received_power(
            distance_m, self.config.environment,
            self.config.tx_gain_dbi, self.config.rx_gain_dbi
        )
        noise_floor_dbm = -174 + 10 * np.log10(self.config.data_rate)
        snr_db = rx_power_dbm - noise_floor_dbm
        noisy_signal = add_awgn(signal, snr_db)
        return noisy_signal, rx_power_dbm
    
    def receive_packet(self, signal: np.ndarray, metadata: dict) -> np.ndarray:
        if metadata['modulation'] == 'DSSS':
            demodulated = self.modulator.despread(signal)
        else:
            demodulated = (signal > 0).astype(int)
        
        if metadata.get('encrypted', False) and self.crypto:
            num_bytes = len(demodulated) // 8
            demodulated_trimmed = demodulated[:num_bytes * 8]
            ciphertext = np.packbits(demodulated_trimmed).tobytes()
            try:
                data = self.crypto.decrypt_array(
                    metadata['iv'], ciphertext,
                    metadata['shape'], metadata['dtype']
                )
            except Exception:
                data = np.zeros(metadata['shape'], dtype=metadata['dtype'])
        else:
            data = demodulated
        return data
    
    def simulate_transmission(self, packet_size_bits: int, distance_m: float, 
                            num_packets: int = 100) -> SimulationResult:
        total_bits_error = 0
        total_packets_error = 0
        rx_powers = []
        
        for _ in range(num_packets):
            data = np.random.randint(0, 2, size=packet_size_bits)
            signal, metadata = self.transmit_packet(data)
            rx_signal, rx_power = self.channel_propagation(signal, distance_m)
            rx_powers.append(rx_power)
            received_data = self.receive_packet(rx_signal, metadata)
            
            if len(received_data) == len(data):
                errors = np.sum(data != received_data)
                total_bits_error += errors
                if errors > 0:
                    total_packets_error += 1
        
        total_bits = packet_size_bits * num_packets
        ber = total_bits_error / total_bits
        per = total_packets_error / num_packets
        avg_rx_power = np.mean(rx_powers)
        link_margin = avg_rx_power - self.config.rx_sensitivity_dbm
        noise_floor_dbm = -174 + 10 * np.log10(self.config.data_rate)
        snr_db = avg_rx_power - noise_floor_dbm
        throughput_kbps = self.config.data_rate / 1000 * (1 - per)
        tx_time_ms = packet_size_bits / self.config.data_rate * 1000
        processing_ms = 5
        latency_ms = tx_time_ms + processing_ms
        
        return SimulationResult(
            ber=ber, per=per, snr_db=snr_db,
            rx_power_dbm=avg_rx_power,
            link_margin_db=link_margin,
            throughput_kbps=throughput_kbps,
            latency_ms=latency_ms
        )

print("=" * 70)
print("TEST RAPIDE - Simulateur Radio Airsoft")
print("=" * 70)

# Configuration pour terrain d'airsoft extérieur
config = RadioConfig(
    use_dsss=True,
    data_rate=10000,
    environment=Environment.OUTDOOR_OPEN,
    encryption_enabled=True,
    password="airsoft2024"
)

print(f"\nConfiguration:")
print(f"  Modulation: DSSS")
print(f"  Débit: {config.data_rate/1000} kbps")
print(f"  Chiffrement: {'Activé' if config.encryption_enabled else 'Désactivé'}")
print(f"  Environnement: {config.environment.value}")

sim = RadioSimulator(config)

# Test à différentes distances typiques pour l'airsoft
test_distances = [100, 300, 500, 1000]

print(f"\nSimulation de transmission (100 paquets de 256 bits):\n")

for distance in test_distances:
    print(f"Distance: {distance}m")
    result = sim.simulate_transmission(
        packet_size_bits=256,
        distance_m=distance,
        num_packets=100
    )
    
    print(f"  ├─ BER: {result.ber:.2e}")
    print(f"  ├─ PER: {result.per:.2%}")
    print(f"  ├─ SNR: {result.snr_db:.1f} dB")
    print(f"  ├─ Puissance reçue: {result.rx_power_dbm:.1f} dBm")
    print(f"  ├─ Marge de liaison: {result.link_margin_db:.1f} dB")
    print(f"  ├─ Débit effectif: {result.throughput_kbps:.2f} kbps")
    print(f"  └─ Latence: {result.latency_ms:.2f} ms")
    
    # Évaluation de la qualité
    if result.link_margin_db > 20:
        quality = "Excellente ✓"
    elif result.link_margin_db > 10:
        quality = "Bonne ✓"
    elif result.link_margin_db > 0:
        quality = "Acceptable ⚠"
    else:
        quality = "Mauvaise ✗"
    
    print(f"     Qualité de liaison: {quality}\n")

print("=" * 70)
print("Test terminé avec succès!")
print("=" * 70)
