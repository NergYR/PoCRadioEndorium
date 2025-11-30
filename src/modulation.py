"""
Module de modulation avec étalement de spectre (DSSS et FHSS)
"""

import numpy as np
from typing import Tuple, List


class DSSSModulator:
    """Modulation par étalement de spectre à séquence directe (Direct Sequence Spread Spectrum)"""
    
    def __init__(self, chip_rate: int = 1000000, data_rate: int = 10000):
        """
        Initialise le modulateur DSSS
        
        Args:
            chip_rate: Taux de chips (Hz) - détermine la bande passante
            data_rate: Taux de données (bits/s)
        """
        self.chip_rate = chip_rate
        self.data_rate = data_rate
        self.spreading_factor = chip_rate // data_rate
        self.pn_sequence = self._generate_pn_sequence()
        
    def _generate_pn_sequence(self) -> np.ndarray:
        """Génère une séquence pseudo-aléatoire pour l'étalement"""
        # Utilise un LFSR (Linear Feedback Shift Register) simplifié
        np.random.seed(42)  # Seed fixe pour la reproductibilité
        return np.random.choice([-1, 1], size=self.spreading_factor)
    
    def spread(self, data: np.ndarray) -> np.ndarray:
        """
        Étale les données avec la séquence PN
        
        Args:
            data: Données binaires à étaler (0 ou 1)
            
        Returns:
            Signal étalé
        """
        # Convertit 0/1 en -1/+1 (forcer un type signé pour éviter le wrap des uint8)
        bipolar_data = 2 * data.astype(np.int16) - 1
        
        # Répète chaque bit selon le facteur d'étalement
        spread_signal = np.repeat(bipolar_data, self.spreading_factor).astype(np.float32)
        
        # Multiplie par la séquence PN répétée
        pn_repeated = np.tile(self.pn_sequence, len(data))
        
        return spread_signal * pn_repeated[:len(spread_signal)]
    
    def despread(self, signal: np.ndarray) -> np.ndarray:
        """
        Désétale le signal reçu
        
        Args:
            signal: Signal étalé reçu
            
        Returns:
            Données récupérées
        """
        # Multiplie par la séquence PN
        pn_repeated = np.tile(self.pn_sequence, len(signal) // self.spreading_factor + 1)
        despread = signal * pn_repeated[:len(signal)]
        
        # Intègre sur chaque période de symbole
        num_symbols = len(signal) // self.spreading_factor
        recovered = np.zeros(num_symbols)
        
        for i in range(num_symbols):
            start = i * self.spreading_factor
            end = start + self.spreading_factor
            recovered[i] = np.sum(despread[start:end])
        
        # Décision binaire
        return (recovered > 0).astype(int)
    
    def get_processing_gain_db(self) -> float:
        """Retourne le gain de traitement en dB"""
        return 10 * np.log10(self.spreading_factor)


class FHSSModulator:
    """Modulation par étalement de spectre à saut de fréquence (Frequency Hopping Spread Spectrum)"""
    
    def __init__(self, num_channels: int = 50, hop_duration: float = 0.1, center_freq: float = 915e6):
        """
        Initialise le modulateur FHSS
        
        Args:
            num_channels: Nombre de canaux de fréquence disponibles
            hop_duration: Durée de chaque saut (secondes)
            center_freq: Fréquence centrale (Hz)
        """
        self.num_channels = num_channels
        self.hop_duration = hop_duration
        self.center_freq = center_freq
        self.channel_spacing = 200e3  # 200 kHz entre canaux
        self.hopping_sequence = self._generate_hopping_sequence()
        
    def _generate_hopping_sequence(self) -> List[int]:
        """Génère une séquence de saut de fréquence pseudo-aléatoire"""
        np.random.seed(42)  # Seed fixe pour émetteur et récepteur
        return np.random.randint(0, self.num_channels, size=1000).tolist()
    
    def get_frequency_at_time(self, time: float) -> float:
        """
        Retourne la fréquence utilisée à un instant donné
        
        Args:
            time: Temps en secondes
            
        Returns:
            Fréquence en Hz
        """
        hop_index = int(time / self.hop_duration) % len(self.hopping_sequence)
        channel = self.hopping_sequence[hop_index]
        return self.center_freq + (channel - self.num_channels / 2) * self.channel_spacing
    
    def modulate(self, data: np.ndarray, duration: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Module les données avec saut de fréquence
        
        Args:
            data: Données à transmettre
            duration: Durée totale de la transmission (secondes)
            
        Returns:
            (temps, fréquences) - séquence de sauts
        """
        num_hops = int(duration / self.hop_duration)
        times = np.arange(num_hops) * self.hop_duration
        frequencies = [self.get_frequency_at_time(t) for t in times]
        
        return times, np.array(frequencies)
    
    def get_hop_rate(self) -> float:
        """Retourne le taux de saut en sauts/seconde"""
        return 1.0 / self.hop_duration


def add_awgn(signal: np.ndarray, snr_db: float) -> np.ndarray:
    """
    Ajoute du bruit blanc gaussien additif (AWGN)
    
    Args:
        signal: Signal à bruiter
        snr_db: Rapport signal/bruit en dB
        
    Returns:
        Signal bruité
    """
    signal_power = np.mean(signal ** 2)
    snr_linear = 10 ** (snr_db / 10)
    noise_power = signal_power / snr_linear
    noise = np.sqrt(noise_power) * np.random.randn(len(signal))
    
    return signal + noise


if __name__ == "__main__":
    # Test DSSS
    print("=== Test DSSS ===")
    dsss = DSSSModulator(chip_rate=1000000, data_rate=10000)
    print(f"Facteur d'étalement: {dsss.spreading_factor}")
    print(f"Gain de traitement: {dsss.get_processing_gain_db():.2f} dB")
    
    # Génère des données aléatoires
    data = np.random.randint(0, 2, size=10)
    print(f"Données originales: {data}")
    
    # Étalement
    spread_signal = dsss.spread(data)
    print(f"Taille du signal étalé: {len(spread_signal)}")
    
    # Ajoute du bruit
    noisy_signal = add_awgn(spread_signal, snr_db=0)
    
    # Désétalement
    recovered_data = dsss.despread(noisy_signal)
    print(f"Données récupérées: {recovered_data}")
    print(f"Erreurs: {np.sum(data != recovered_data)}/{len(data)}")
    
    # Test FHSS
    print("\n=== Test FHSS ===")
    fhss = FHSSModulator(num_channels=50, hop_duration=0.1)
    print(f"Taux de saut: {fhss.get_hop_rate()} sauts/s")
    
    times, freqs = fhss.modulate(data, duration=1.0)
    print(f"Nombre de sauts: {len(times)}")
    print(f"Plage de fréquences: {freqs.min()/1e6:.2f} - {freqs.max()/1e6:.2f} MHz")
