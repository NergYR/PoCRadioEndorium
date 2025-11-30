"""
Simulateur complet de système radio pour airsoft
"""

import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import List, Tuple, Optional
from .modulation import DSSSModulator, FHSSModulator, add_awgn
from .crypto import RadioCrypto, MessageAuthenticator
from .propagation import PropagationModel, Environment
from .doppler import DopplerChannel, MobileScenario


@dataclass
class RadioConfig:
    """Configuration du système radio"""
    # Modulation
    chip_rate: int = 1000000  # 1 Mchip/s
    data_rate: int = 10000    # 10 kbps
    use_dsss: bool = True     # True=DSSS, False=FHSS
    
    # FHSS parameters
    num_channels: int = 50
    hop_duration: float = 0.1
    
    # RF parameters
    frequency_mhz: float = 915.0
    tx_power_dbm: float = 20.0
    rx_sensitivity_dbm: float = -110.0
    tx_gain_dbi: float = 2.0
    rx_gain_dbi: float = 2.0
    
    # Crypto
    encryption_enabled: bool = True
    password: str = "airsoft2024"
    
    # Environment
    environment: Environment = Environment.OUTDOOR_OPEN
    
    # Mobility (Doppler)
    mobile_scenario: Optional[MobileScenario] = None
    enable_rayleigh_fading: bool = False


@dataclass
class SimulationResult:
    """Résultats de simulation"""
    ber: float  # Bit Error Rate
    per: float  # Packet Error Rate
    snr_db: float
    rx_power_dbm: float
    link_margin_db: float
    throughput_kbps: float
    latency_ms: float


class RadioSimulator:
    """Simulateur de système radio complet"""
    
    def __init__(self, config: RadioConfig):
        """
        Initialise le simulateur
        
        Args:
            config: Configuration du système
        """
        self.config = config
        
        # Initialise les modules
        if config.use_dsss:
            self.modulator = DSSSModulator(config.chip_rate, config.data_rate)
        else:
            self.modulator = FHSSModulator(config.num_channels, config.hop_duration, 
                                          config.frequency_mhz * 1e6)
        
        self.crypto = RadioCrypto(config.password) if config.encryption_enabled else None
        self.authenticator = MessageAuthenticator() if config.encryption_enabled else None
        self.propagation = PropagationModel(config.frequency_mhz, config.tx_power_dbm)
        
        # Canal Doppler
        self.doppler = DopplerChannel(carrier_freq_hz=config.frequency_mhz * 1e6) if config.mobile_scenario else None
    
    def transmit_packet(self, data: np.ndarray) -> Tuple[np.ndarray, dict]:
        """
        Simule la transmission d'un paquet
        
        Args:
            data: Données binaires à transmettre
            
        Returns:
            (signal_transmis, metadata)
        """
        metadata = {}
        
        # Chiffrement: pack des bits -> bytes, AES-CTR, HMAC
        if self.crypto:
            packed_plain = np.packbits(data.astype(np.uint8), bitorder='little')
            iv, ciphertext, mac = self.crypto.encrypt(packed_plain.tobytes())
            # Convertit le ciphertext en bits pour la chaîne de transmission
            data_to_transmit = np.unpackbits(np.frombuffer(ciphertext, dtype=np.uint8), bitorder='little')
            metadata['iv'] = iv
            metadata['mac'] = mac
            metadata['bit_len'] = int(len(data))
            metadata['encrypted'] = True
        else:
            data_to_transmit = data
            metadata['encrypted'] = False
        
        # Modulation avec étalement de spectre
        if self.config.use_dsss:
            signal = self.modulator.spread(data_to_transmit)
            metadata['modulation'] = 'DSSS'
            metadata['spreading_factor'] = self.modulator.spreading_factor
        else:
            # FHSS: mappe les bits en -1/+1 pour une transmission robuste au bruit
            signal = (2 * data_to_transmit.astype(np.int16) - 1).astype(np.float32)
            metadata['modulation'] = 'FHSS'
            metadata['hop_rate'] = self.modulator.get_hop_rate()
        
        metadata['signal_length'] = len(signal)
        
        return signal, metadata
    
    def channel_propagation(self, signal: np.ndarray, distance_m: float, 
                           time_offset_s: float = 0.0) -> Tuple[np.ndarray, float]:
        """
        Simule la propagation dans le canal radio avec effet Doppler optionnel
        
        Args:
            signal: Signal à transmettre
            distance_m: Distance de transmission
            time_offset_s: Temps écoulé depuis le début du scénario mobile (pour Doppler)
            
        Returns:
            (signal_reçu, puissance_reçue_dbm)
        """
        # Calcule la puissance reçue
        rx_power_dbm = self.propagation.received_power(
            distance_m,
            self.config.environment,
            self.config.tx_gain_dbi,
            self.config.rx_gain_dbi
        )
        
        # Calcule le SNR
        noise_floor_dbm = -174 + 10 * np.log10(self.config.data_rate)  # Bruit thermique
        snr_db = rx_power_dbm - noise_floor_dbm
        
        # Applique l'effet Doppler si configuré
        processed_signal = signal
        if self.doppler and self.config.mobile_scenario:
            # Calcule le décalage Doppler instantané
            scenario = self.config.mobile_scenario
            
            # Position à l'instant time_offset_s
            angle_rad = np.deg2rad(scenario.angle_deg)
            vx = scenario.velocity_ms * np.cos(angle_rad)
            vy = scenario.velocity_ms * np.sin(angle_rad)
            
            x0 = scenario.distance_m * np.cos(angle_rad)
            y0 = scenario.distance_m * np.sin(angle_rad)
            
            x = x0 + vx * time_offset_s
            y = y0 + vy * time_offset_s
            
            current_distance = np.sqrt(x**2 + y**2)
            
            # Angle instantané par rapport à la ligne de vue
            if current_distance > 0:
                current_angle = np.arctan2(y, x)
                radial_velocity = scenario.velocity_ms * np.cos(current_angle - angle_rad)
            else:
                radial_velocity = 0
            
            # Calcule le décalage Doppler
            doppler_shift = (radial_velocity / 3e8) * (self.config.frequency_mhz * 1e6)
            
            # Applique le décalage (si signal est long, prendre taux d'échantillonnage)
            # Pour simplification, on utilise chip_rate comme taux d'échantillonnage
            sample_rate = self.config.chip_rate
            processed_signal = self.doppler.apply_doppler_to_signal(
                signal.astype(complex) if not np.iscomplexobj(signal) else signal,
                sample_rate,
                doppler_shift
            )
            
            # Évanouissement de Rayleigh si activé
            if self.config.enable_rayleigh_fading:
                doppler_spread = self.doppler.calculate_doppler_spread(scenario.velocity_ms)
                processed_signal = self.doppler.simulate_rayleigh_fading(
                    processed_signal,
                    doppler_spread,
                    sample_rate
                )
            
            # Reconvertit en réel pour la suite du traitement
            if np.iscomplexobj(processed_signal):
                processed_signal = np.real(processed_signal)
        
        # Ajoute du bruit AWGN
        noisy_signal = add_awgn(processed_signal, snr_db)
        
        return noisy_signal, rx_power_dbm
    
    def receive_packet(self, signal: np.ndarray, metadata: dict) -> np.ndarray:
        """
        Simule la réception d'un paquet
        
        Args:
            signal: Signal reçu
            metadata: Métadonnées de transmission
            
        Returns:
            Données décodées
        """
        # Démodulation
        if metadata['modulation'] == 'DSSS':
            demodulated = self.modulator.despread(signal)
        else:
            # Démodule le signal bipolarisé en bits 0/1
            demodulated = (signal > 0).astype(np.uint8)
        
        # Déchiffrement
        if metadata.get('encrypted', False) and self.crypto:
            # Convertit les bits reçus en bytes (ciphertext)
            num_bytes = len(demodulated) // 8
            demodulated_trimmed = demodulated[:num_bytes * 8]
            ciphertext = np.packbits(demodulated_trimmed.astype(np.uint8), bitorder='little').tobytes()

            try:
                # Déchiffre puis reconstruit les bits originaux
                plaintext_bytes = self.crypto.decrypt(
                    metadata['iv'], ciphertext, metadata.get('mac', b'')
                )
                plaintext_bits = np.unpackbits(np.frombuffer(plaintext_bytes, dtype=np.uint8), bitorder='little')
                bit_len = int(metadata.get('bit_len', len(plaintext_bits)))
                data = plaintext_bits[:bit_len]
            except Exception:
                # Erreur de déchiffrement (intégrité/erreurs de bits)
                bit_len = int(metadata.get('bit_len', 0))
                data = np.zeros(bit_len, dtype=np.uint8)
        else:
            data = demodulated

        return data
    
    def simulate_transmission(self, packet_size_bits: int, distance_m: float, 
                            num_packets: int = 100, time_offset_s: float = 0.0) -> SimulationResult:
        """
        Simule la transmission de plusieurs paquets
        
        Args:
            packet_size_bits: Taille de chaque paquet en bits
            distance_m: Distance de transmission
            num_packets: Nombre de paquets à simuler
            time_offset_s: Temps écoulé (pour simulations Doppler)
            
        Returns:
            Résultats de la simulation
        """
        total_bits_error = 0
        total_packets_error = 0
        rx_powers = []
        
        for _ in range(num_packets):
            # Génère un paquet aléatoire
            data = np.random.randint(0, 2, size=packet_size_bits)
            
            # Transmission
            signal, metadata = self.transmit_packet(data)
            
            # Propagation avec Doppler
            rx_signal, rx_power = self.channel_propagation(signal, distance_m, time_offset_s)
            rx_powers.append(rx_power)
            
            # Réception
            received_data = self.receive_packet(rx_signal, metadata)
            
            # Calcule les erreurs
            if len(received_data) == len(data):
                errors = np.sum(data != received_data)
                total_bits_error += errors
                
                if errors > 0:
                    total_packets_error += 1
        
        # Calcule les métriques
        total_bits = packet_size_bits * num_packets
        ber = total_bits_error / total_bits
        per = total_packets_error / num_packets
        
        avg_rx_power = np.mean(rx_powers)
        link_margin = avg_rx_power - self.config.rx_sensitivity_dbm
        
        # Calcule le SNR moyen
        noise_floor_dbm = -174 + 10 * np.log10(self.config.data_rate)
        snr_db = avg_rx_power - noise_floor_dbm
        
        # Débit effectif (tenant compte du PER)
        throughput_kbps = self.config.data_rate / 1000 * (1 - per)
        
        # Latence (temps de transmission + traitement)
        tx_time_ms = packet_size_bits / self.config.data_rate * 1000
        processing_ms = 5  # Temps de traitement estimé
        latency_ms = tx_time_ms + processing_ms
        
        return SimulationResult(
            ber=ber,
            per=per,
            snr_db=snr_db,
            rx_power_dbm=avg_rx_power,
            link_margin_db=link_margin,
            throughput_kbps=throughput_kbps,
            latency_ms=latency_ms
        )
    
    def plot_range_analysis(self, max_distance_m: int = 2000, step_m: int = 50):
        """
        Génère des graphiques d'analyse de portée
        
        Args:
            max_distance_m: Distance maximale à tester
            step_m: Pas de distance
        """
        distances = np.arange(10, max_distance_m, step_m)
        bers = []
        rx_powers = []
        snrs = []
        
        print("Simulation en cours...")
        for dist in distances:
            result = self.simulate_transmission(100, dist, num_packets=50)
            bers.append(result.ber)
            rx_powers.append(result.rx_power_dbm)
            snrs.append(result.snr_db)
        
        # Crée les graphiques
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle(f'Analyse de performance - {self.config.environment.value}', fontsize=14)
        
        # BER vs Distance
        axes[0, 0].semilogy(distances, bers, 'b-', linewidth=2)
        axes[0, 0].set_xlabel('Distance (m)')
        axes[0, 0].set_ylabel('BER')
        axes[0, 0].set_title('Taux d\'erreur binaire')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Puissance reçue vs Distance
        axes[0, 1].plot(distances, rx_powers, 'r-', linewidth=2)
        axes[0, 1].axhline(y=self.config.rx_sensitivity_dbm, color='k', linestyle='--', 
                          label='Sensibilité Rx')
        axes[0, 1].set_xlabel('Distance (m)')
        axes[0, 1].set_ylabel('Puissance reçue (dBm)')
        axes[0, 1].set_title('Puissance du signal reçu')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # SNR vs Distance
        axes[1, 0].plot(distances, snrs, 'g-', linewidth=2)
        axes[1, 0].set_xlabel('Distance (m)')
        axes[1, 0].set_ylabel('SNR (dB)')
        axes[1, 0].set_title('Rapport signal/bruit')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Zone de couverture
        valid_distances = [d for d, ber in zip(distances, bers) if ber < 0.001]
        max_range = max(valid_distances) if valid_distances else 0
        
        axes[1, 1].text(0.5, 0.7, f'Portée maximale estimée:\n{max_range:.0f} m',
                       ha='center', va='center', fontsize=16, 
                       bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        axes[1, 1].text(0.5, 0.3, f'Configuration:\n'
                                  f'Modulation: {self.config.use_dsss and "DSSS" or "FHSS"}\n'
                                  f'Débit: {self.config.data_rate/1000:.1f} kbps\n'
                                  f'Chiffrement: {"Oui" if self.config.encryption_enabled else "Non"}',
                       ha='center', va='center', fontsize=12,
                       bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        plt.savefig('range_analysis.png', dpi=150)
        print("Graphique sauvegardé: range_analysis.png")
        plt.show()


if __name__ == "__main__":
    print("=== Simulateur Radio Airsoft ===\n")
    
    # Configuration pour terrain extérieur
    config_outdoor = RadioConfig(
        use_dsss=True,
        environment=Environment.OUTDOOR_OPEN,
        encryption_enabled=True
    )
    
    sim = RadioSimulator(config_outdoor)
    
    # Test à différentes distances
    test_distances = [50, 100, 200, 500, 1000]
    
    print("Configuration:")
    print(f"  Modulation: {'DSSS' if config_outdoor.use_dsss else 'FHSS'}")
    print(f"  Débit: {config_outdoor.data_rate/1000} kbps")
    print(f"  Chiffrement: {'Activé' if config_outdoor.encryption_enabled else 'Désactivé'}")
    print(f"  Environnement: {config_outdoor.environment.value}")
    print(f"\nTests de transmission:\n")
    
    for distance in test_distances:
        result = sim.simulate_transmission(packet_size_bits=128, distance_m=distance, num_packets=100)
        print(f"Distance: {distance}m")
        print(f"  BER: {result.ber:.2e}")
        print(f"  PER: {result.per:.2%}")
        print(f"  SNR: {result.snr_db:.1f} dB")
        print(f"  Puissance reçue: {result.rx_power_dbm:.1f} dBm")
        print(f"  Marge de liaison: {result.link_margin_db:.1f} dB")
        print(f"  Débit effectif: {result.throughput_kbps:.2f} kbps")
        print(f"  Latence: {result.latency_ms:.2f} ms")
        print()
    
    # Génère l'analyse de portée
    print("\nGénération de l'analyse de portée...")
    sim.plot_range_analysis(max_distance_m=2000, step_m=100)
