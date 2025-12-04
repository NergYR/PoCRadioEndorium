"""
Système de contrôle adaptatif pour compensation Doppler et gestion du bruit
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Dict
from enum import Enum
from collections import deque


class AdaptiveMode(Enum):
    """Modes de fonctionnement adaptatif"""
    MANUAL = "manual"
    AUTO_POWER = "auto_power"           # Ajuste puissance TX
    AUTO_RATE = "auto_rate"             # Ajuste débit
    AUTO_MODULATION = "auto_modulation" # Change DSSS/FHSS
    FULL_AUTO = "full_auto"             # Tous les paramètres


@dataclass
class ChannelEstimate:
    """Estimation des conditions du canal"""
    snr_db: float
    doppler_shift_hz: float
    doppler_spread_hz: float
    ber: float
    per: float
    rx_power_dbm: float
    noise_floor_dbm: float
    timestamp: float = 0.0
    
    @property
    def is_good(self) -> bool:
        """Canal de bonne qualité"""
        return self.snr_db > 15 and self.ber < 1e-3
    
    @property
    def is_degraded(self) -> bool:
        """Canal dégradé"""
        return 10 < self.snr_db <= 15 or 1e-3 <= self.ber < 1e-2
    
    @property
    def is_poor(self) -> bool:
        """Canal de mauvaise qualité"""
        return self.snr_db <= 10 or self.ber >= 1e-2
    
    def quality_score(self) -> float:
        """Score de qualité global du canal (0-100)"""
        # SNR contribution (40% du score)
        snr_score = min(100, max(0, (self.snr_db + 10) / 0.4))  # -10 dB = 0, 30 dB = 100
        
        # BER contribution (30% du score)
        if self.ber == 0:
            ber_score = 100
        else:
            ber_score = min(100, max(0, 100 + 20 * np.log10(self.ber)))  # 1e-5 = 100, 1e-1 = 0
        
        # Doppler contribution (20% du score)
        doppler_score = min(100, max(0, 100 - self.doppler_spread_hz / 2))  # 0 Hz = 100, 200 Hz = 0
        
        # Puissance reçue contribution (10% du score)
        sensitivity = -110  # dBm
        power_margin = self.rx_power_dbm - sensitivity
        power_score = min(100, max(0, power_margin * 2.5))  # 0 dB = 0, 40 dB = 100
        
        # Score pondéré
        total_score = (snr_score * 0.4 + ber_score * 0.3 + 
                      doppler_score * 0.2 + power_score * 0.1)
        
        return total_score


@dataclass
class AdaptiveParameters:
    """Paramètres adaptatifs du système"""
    tx_power_dbm: float = 20.0
    data_rate: int = 10000
    spreading_factor: int = 100
    use_dsss: bool = True
    fec_enabled: bool = False
    fec_redundancy: float = 0.5  # 0.0 à 1.0 (0% à 100% de redondance)
    interleaving_enabled: bool = False
    interleaving_depth: int = 8  # Profondeur d'entrelacement (2, 4, 8, 16)
    
    # Compensation Doppler
    doppler_compensation_enabled: bool = True
    frequency_offset_hz: float = 0.0
    
    # Seuils de décision
    snr_target_db: float = 20.0
    ber_target: float = 1e-4
    quality_threshold: float = 60.0  # Score de qualité minimum acceptable


class AdaptiveController:
    """Contrôleur adaptatif pour optimisation temps réel"""
    
    def __init__(self, mode: AdaptiveMode = AdaptiveMode.FULL_AUTO, 
                 prediction_enabled: bool = True,
                 history_size: int = 100):
        """
        Initialise le contrôleur adaptatif
        
        Args:
            mode: Mode de fonctionnement adaptatif
            prediction_enabled: Active la prédiction des dégradations
            history_size: Taille de l'historique pour prédiction
        """
        self.mode = mode
        self.params = AdaptiveParameters()
        self.history = []
        self.prediction_enabled = prediction_enabled
        
        # Buffers circulaires pour prédiction
        self.snr_buffer = deque(maxlen=history_size)
        self.ber_buffer = deque(maxlen=history_size)
        self.doppler_buffer = deque(maxlen=history_size)
        self.quality_buffer = deque(maxlen=history_size)
        
        # Limites système
        self.tx_power_min = 0.0   # dBm
        self.tx_power_max = 27.0  # dBm (légal pour 915 MHz)
        self.data_rate_options = [5000, 10000, 20000, 50000]  # bps
        self.sf_options = [50, 100, 200, 400]  # Spreading factors
        self.fec_redundancy_options = [0.0, 0.25, 0.5, 0.75]  # Taux de redondance FEC
        self.interleaving_depths = [2, 4, 8, 16]  # Profondeurs d'entrelacement
        
    def predict_degradation(self) -> Dict[str, float]:
        """
        Prédit la dégradation future du canal basée sur l'historique
        
        Returns:
            Prédictions: {snr_trend, ber_trend, doppler_trend, quality_trend}
        """
        if not self.prediction_enabled or len(self.snr_buffer) < 10:
            return {
                'snr_trend': 0.0,
                'ber_trend': 0.0,
                'doppler_trend': 0.0,
                'quality_trend': 0.0,
                'degradation_probability': 0.0
            }
        
        # Calcule les tendances par régression linéaire simple
        n = min(20, len(self.snr_buffer))  # Utilise les 20 derniers échantillons
        x = np.arange(n)
        
        # Tendance SNR
        snr_recent = list(self.snr_buffer)[-n:]
        snr_trend = np.polyfit(x, snr_recent, 1)[0]  # Pente
        
        # Tendance BER (log scale)
        ber_recent = list(self.ber_buffer)[-n:]
        ber_log = [np.log10(b + 1e-10) for b in ber_recent]
        ber_trend = np.polyfit(x, ber_log, 1)[0]
        
        # Tendance Doppler
        doppler_recent = list(self.doppler_buffer)[-n:]
        doppler_trend = np.polyfit(x, doppler_recent, 1)[0]
        
        # Tendance qualité globale
        quality_recent = list(self.quality_buffer)[-n:]
        quality_trend = np.polyfit(x, quality_recent, 1)[0]
        
        # Probabilité de dégradation (basée sur tendances négatives)
        degradation_indicators = 0
        if snr_trend < -0.5:  # SNR diminue rapidement
            degradation_indicators += 1
        if ber_trend > 0.1:   # BER augmente
            degradation_indicators += 1
        if abs(doppler_trend) > 1.0:  # Doppler change rapidement
            degradation_indicators += 1
        if quality_trend < -1.0:  # Qualité diminue
            degradation_indicators += 1
        
        degradation_probability = degradation_indicators / 4.0
        
        return {
            'snr_trend': float(snr_trend),
            'ber_trend': float(ber_trend),
            'doppler_trend': float(doppler_trend),
            'quality_trend': float(quality_trend),
            'degradation_probability': degradation_probability
        }
    
    def estimate_channel(self, signal_rx: np.ndarray, 
                        signal_tx: np.ndarray,
                        rx_power_dbm: float,
                        noise_floor_dbm: float,
                        time_vector: Optional[np.ndarray] = None,
                        timestamp: float = 0.0) -> ChannelEstimate:
        """
        Estime les conditions actuelles du canal
        
        Args:
            signal_rx: Signal reçu
            signal_tx: Signal transmis (référence)
            rx_power_dbm: Puissance reçue
            noise_floor_dbm: Niveau de bruit
            time_vector: Vecteur temporel pour estimation Doppler
            
        Returns:
            Estimation des conditions du canal
        """
        # Calcul du SNR
        snr_db = rx_power_dbm - noise_floor_dbm
        
        # Estimation du BER (bits erronés / total)
        if len(signal_rx) == len(signal_tx):
            errors = np.sum(signal_rx != signal_tx)
            ber = errors / len(signal_tx) if len(signal_tx) > 0 else 0.0
        else:
            ber = 0.5  # Pire cas si longueurs différentes
        
        # Estimation du PER (paquets avec au moins 1 erreur)
        packet_size = 256
        num_packets = len(signal_tx) // packet_size
        packets_error = 0
        for i in range(num_packets):
            start = i * packet_size
            end = start + packet_size
            if np.any(signal_rx[start:end] != signal_tx[start:end]):
                packets_error += 1
        per = packets_error / num_packets if num_packets > 0 else 0.0
        
        # Estimation du Doppler via FFT du signal
        if time_vector is not None and len(signal_rx) > 100:
            # Transformée de Fourier pour détecter le décalage
            fft = np.fft.fft(signal_rx.astype(complex) if not np.iscomplexobj(signal_rx) else signal_rx)
            freqs = np.fft.fftfreq(len(signal_rx), time_vector[1] - time_vector[0] if len(time_vector) > 1 else 1e-6)
            
            # Pic de puissance = décalage Doppler approximatif
            power = np.abs(fft)**2
            peak_idx = np.argmax(power[1:len(power)//2]) + 1
            doppler_shift = freqs[peak_idx]
            
            # Étalement = largeur du spectre
            half_power = np.max(power) / 2
            indices_above = np.where(power > half_power)[0]
            if len(indices_above) > 1:
                doppler_spread = (freqs[indices_above[-1]] - freqs[indices_above[0]])
            else:
                doppler_spread = 0.0
        else:
            doppler_shift = 0.0
            doppler_spread = 0.0
        
        return ChannelEstimate(
            snr_db=snr_db,
            doppler_shift_hz=doppler_shift,
            doppler_spread_hz=abs(doppler_spread),
            ber=ber,
            per=per,
            rx_power_dbm=rx_power_dbm,
            noise_floor_dbm=noise_floor_dbm,
            timestamp=timestamp
        )
    
    def compensate_doppler(self, signal: np.ndarray, 
                          doppler_shift_hz: float,
                          sample_rate: float) -> np.ndarray:
        """
        Compense le décalage Doppler dans le signal
        
        Args:
            signal: Signal à compenser
            doppler_shift_hz: Décalage Doppler estimé
            sample_rate: Taux d'échantillonnage
            
        Returns:
            Signal compensé
        """
        if not self.params.doppler_compensation_enabled:
            return signal
        
        # Applique un décalage de fréquence inverse
        t = np.arange(len(signal)) / sample_rate
        phase_shift = -2 * np.pi * doppler_shift_hz * t
        
        if np.iscomplexobj(signal):
            compensated = signal * np.exp(1j * phase_shift)
        else:
            # Pour signal réel, applique via multiplication complexe puis partie réelle
            compensated = np.real(signal.astype(complex) * np.exp(1j * phase_shift))
        
        return compensated
    
    def adapt_tx_power(self, channel: ChannelEstimate) -> float:
        """
        Ajuste la puissance d'émission selon les conditions
        
        Args:
            channel: Estimation du canal
            
        Returns:
            Nouvelle puissance TX en dBm
        """
        if self.mode == AdaptiveMode.MANUAL:
            return self.params.tx_power_dbm
        
        # Calcul de l'erreur par rapport au SNR cible
        snr_error = self.params.snr_target_db - channel.snr_db
        
        # Contrôle proportionnel (gain = 0.5 pour éviter oscillations)
        power_adjustment = 0.5 * snr_error
        
        # Nouvelle puissance
        new_power = self.params.tx_power_dbm + power_adjustment
        
        # Limite aux valeurs autorisées
        new_power = np.clip(new_power, self.tx_power_min, self.tx_power_max)
        
        return new_power
    
    def adapt_data_rate(self, channel: ChannelEstimate) -> int:
        """
        Ajuste le débit de données selon les conditions
        
        Args:
            channel: Estimation du canal
            
        Returns:
            Nouveau débit en bps
        """
        if self.mode == AdaptiveMode.MANUAL:
            return self.params.data_rate
        
        current_idx = self.data_rate_options.index(self.params.data_rate) if self.params.data_rate in self.data_rate_options else 1
        
        # Conditions bonnes : augmente le débit
        if channel.is_good and current_idx < len(self.data_rate_options) - 1:
            return self.data_rate_options[current_idx + 1]
        
        # Conditions dégradées : réduit le débit
        elif channel.is_degraded and current_idx > 0:
            return self.data_rate_options[current_idx - 1]
        
        # Conditions très mauvaises : débit minimum
        elif channel.is_poor:
            return self.data_rate_options[0]
        
        # Sinon, garde le débit actuel
        return self.params.data_rate
    
    def adapt_spreading_factor(self, channel: ChannelEstimate) -> int:
        """
        Ajuste le facteur d'étalement selon les conditions
        
        Args:
            channel: Estimation du canal
            
        Returns:
            Nouveau spreading factor
        """
        if self.mode == AdaptiveMode.MANUAL:
            return self.params.spreading_factor
        
        current_idx = self.sf_options.index(self.params.spreading_factor) if self.params.spreading_factor in self.sf_options else 1
        
        # Doppler élevé ou bruit fort : augmente SF pour plus de robustesse
        if (channel.doppler_spread_hz > 50 or channel.snr_db < 15) and current_idx < len(self.sf_options) - 1:
            return self.sf_options[current_idx + 1]
        
        # Conditions excellentes : réduit SF pour augmenter débit effectif
        elif channel.is_good and channel.doppler_spread_hz < 20 and current_idx > 0:
            return self.sf_options[current_idx - 1]
        
        return self.params.spreading_factor
    
    def adapt_fec(self, channel: ChannelEstimate, prediction: Dict[str, float]) -> Tuple[bool, float]:
        """
        Adapte le codage correcteur d'erreurs selon les conditions
        
        Args:
            channel: Estimation du canal
            prediction: Prédictions de dégradation
            
        Returns:
            (fec_enabled, redundancy_rate)
        """
        if self.mode == AdaptiveMode.MANUAL:
            return self.params.fec_enabled, self.params.fec_redundancy
        
        quality = channel.quality_score()
        
        # Conditions excellentes : pas de FEC (gain de débit)
        if quality > 80 and prediction['degradation_probability'] < 0.3:
            return False, 0.0
        
        # Conditions bonnes : FEC léger
        elif quality > 60:
            return True, 0.25
        
        # Conditions moyennes : FEC moyen
        elif quality > 40:
            return True, 0.5
        
        # Conditions dégradées : FEC fort
        else:
            return True, 0.75
    
    def adapt_interleaving(self, channel: ChannelEstimate) -> Tuple[bool, int]:
        """
        Adapte l'entrelacement selon la mobilité (erreurs en rafale)
        
        Args:
            channel: Estimation du canal
            
        Returns:
            (interleaving_enabled, depth)
        """
        if self.mode == AdaptiveMode.MANUAL:
            return self.params.interleaving_enabled, self.params.interleaving_depth
        
        # Doppler élevé ou PER élevé → erreurs en rafale → entrelacement
        if channel.doppler_spread_hz > 50 or channel.per > 0.05:
            # Plus le Doppler est élevé, plus la profondeur augmente
            if channel.doppler_spread_hz > 150:
                depth = 16
            elif channel.doppler_spread_hz > 100:
                depth = 8
            elif channel.doppler_spread_hz > 50:
                depth = 4
            else:
                depth = 2
            
            return True, depth
        
        # Conditions stables : pas d'entrelacement (moins de latence)
        return False, 2
    
    def adapt_modulation(self, channel: ChannelEstimate) -> bool:
        """
        Sélectionne DSSS ou FHSS selon les conditions
        
        Args:
            channel: Estimation du canal
            
        Returns:
            True pour DSSS, False pour FHSS
        """
        if self.mode == AdaptiveMode.MANUAL:
            return self.params.use_dsss
        
        # DSSS : Meilleur pour Doppler faible, haute interférence
        # FHSS : Meilleur pour Doppler élevé, interférence localisée
        
        if channel.doppler_spread_hz > 100:
            # Doppler élevé : FHSS évite la dégradation sur une seule fréquence
            return False
        elif channel.snr_db < 10:
            # SNR faible : DSSS avec gain de traitement élevé
            return True
        else:
            # Par défaut, garde DSSS
            return True
    
    def update(self, channel: ChannelEstimate) -> AdaptiveParameters:
        """
        Met à jour tous les paramètres adaptatifs
        
        Args:
            channel: Estimation actuelle du canal
            
        Returns:
            Nouveaux paramètres adaptatifs
        """
        # Sauvegarde l'historique
        self.history.append((channel, self.params))
        
        # Met à jour les buffers pour prédiction
        self.snr_buffer.append(channel.snr_db)
        self.ber_buffer.append(channel.ber)
        self.doppler_buffer.append(abs(channel.doppler_shift_hz))
        self.quality_buffer.append(channel.quality_score())
        
        # Prédiction des dégradations
        prediction = self.predict_degradation()
        
        # Mise à jour selon le mode
        if self.mode == AdaptiveMode.MANUAL:
            return self.params
        
        # Compensation Doppler (toujours active si possible)
        if self.params.doppler_compensation_enabled:
            self.params.frequency_offset_hz = channel.doppler_shift_hz
        
        # Adaptation de la puissance (avec anticipation)
        if self.mode in [AdaptiveMode.AUTO_POWER, AdaptiveMode.FULL_AUTO]:
            base_power = self.adapt_tx_power(channel)
            # Augmente préventivement si dégradation prédite
            if prediction['degradation_probability'] > 0.6:
                base_power = min(base_power + 3.0, self.tx_power_max)
            self.params.tx_power_dbm = base_power
        
        # Adaptation du débit
        if self.mode in [AdaptiveMode.AUTO_RATE, AdaptiveMode.FULL_AUTO]:
            self.params.data_rate = self.adapt_data_rate(channel)
        
        # Adaptation du spreading factor et modulation
        if self.mode in [AdaptiveMode.AUTO_MODULATION, AdaptiveMode.FULL_AUTO]:
            self.params.spreading_factor = self.adapt_spreading_factor(channel)
            self.params.use_dsss = self.adapt_modulation(channel)
        
        # Adaptation FEC et entrelacement (mode FULL_AUTO uniquement)
        if self.mode == AdaptiveMode.FULL_AUTO:
            self.params.fec_enabled, self.params.fec_redundancy = self.adapt_fec(channel, prediction)
            self.params.interleaving_enabled, self.params.interleaving_depth = self.adapt_interleaving(channel)
        
        return self.params
    
    def get_statistics(self) -> dict:
        """
        Retourne les statistiques d'adaptation
        
        Returns:
            Dictionnaire avec les stats
        """
        if not self.history:
            return {}
        
        channels = [h[0] for h in self.history]
        params_list = [h[1] for h in self.history]
        
        # Statistiques basiques
        stats = {
            'num_adaptations': len(self.history),
            'avg_snr_db': np.mean([c.snr_db for c in channels]),
            'avg_ber': np.mean([c.ber for c in channels]),
            'avg_doppler_hz': np.mean([abs(c.doppler_shift_hz) for c in channels]),
            'avg_quality_score': np.mean([c.quality_score() for c in channels]),
            'power_changes': len(set([h[1].tx_power_dbm for h in self.history])),
            'rate_changes': len(set([h[1].data_rate for h in self.history])),
            'modulation_changes': len(set([h[1].use_dsss for h in self.history]))
        }
        
        # Statistiques FEC et entrelacement
        fec_active_count = sum(1 for p in params_list if p.fec_enabled)
        interleaving_active_count = sum(1 for p in params_list if p.interleaving_enabled)
        
        stats.update({
            'fec_usage_percent': (fec_active_count / len(params_list)) * 100 if params_list else 0,
            'interleaving_usage_percent': (interleaving_active_count / len(params_list)) * 100 if params_list else 0,
            'avg_fec_redundancy': np.mean([p.fec_redundancy for p in params_list if p.fec_enabled]) if fec_active_count > 0 else 0,
            'avg_interleaving_depth': np.mean([p.interleaving_depth for p in params_list if p.interleaving_enabled]) if interleaving_active_count > 0 else 0
        })
        
        # Prédiction (si disponible)
        if self.prediction_enabled and len(self.quality_buffer) >= 10:
            prediction = self.predict_degradation()
            stats['current_degradation_probability'] = prediction['degradation_probability']
            stats['quality_trend'] = prediction['quality_trend']
        
        return stats


if __name__ == "__main__":
    print("=== Test du Contrôleur Adaptatif ===\n")
    
    # Test 1: Estimation de canal avec bruit variable
    print("--- Test 1: Estimation de canal ---")
    controller = AdaptiveController(mode=AdaptiveMode.FULL_AUTO)
    
    # Simule un signal avec erreurs
    signal_tx = np.random.randint(0, 2, size=1000)
    signal_rx = signal_tx.copy()
    signal_rx[:50] = 1 - signal_rx[:50]  # 5% d'erreurs
    
    channel = controller.estimate_channel(
        signal_rx=signal_rx,
        signal_tx=signal_tx,
        rx_power_dbm=-90,
        noise_floor_dbm=-110
    )
    
    print(f"SNR: {channel.snr_db:.1f} dB")
    print(f"BER: {channel.ber:.2e}")
    print(f"PER: {channel.per:.2%}")
    print(f"État: {'Bon' if channel.is_good else 'Dégradé' if channel.is_degraded else 'Mauvais'}")
    
    # Test 2: Adaptation de puissance
    print("\n--- Test 2: Adaptation de puissance ---")
    scenarios = [
        ("SNR élevé", -80, -110),  # SNR = 30 dB
        ("SNR moyen", -95, -110),  # SNR = 15 dB
        ("SNR faible", -105, -110) # SNR = 5 dB
    ]
    
    for name, rx_power, noise in scenarios:
        signal_rx = signal_tx.copy()  # Sans erreurs
        channel = controller.estimate_channel(signal_rx, signal_tx, rx_power, noise)
        new_power = controller.adapt_tx_power(channel)
        print(f"{name}: SNR={channel.snr_db:.1f} dB → TX power: {controller.params.tx_power_dbm:.1f} → {new_power:.1f} dBm")
    
    # Test 3: Adaptation du débit
    print("\n--- Test 3: Adaptation du débit ---")
    controller.params.data_rate = 10000
    
    for name, rx_power, noise in scenarios:
        signal_rx = signal_tx.copy()
        channel = controller.estimate_channel(signal_rx, signal_tx, rx_power, noise)
        new_rate = controller.adapt_data_rate(channel)
        print(f"{name}: SNR={channel.snr_db:.1f} dB → Débit: {controller.params.data_rate} → {new_rate} bps")
    
    # Test 4: Compensation Doppler
    print("\n--- Test 4: Compensation Doppler ---")
    sample_rate = 1e6
    doppler_shift = 50  # Hz
    signal = np.random.randn(1000)
    t = np.arange(len(signal)) / sample_rate
    
    # Applique un décalage Doppler
    signal_doppler = signal * np.cos(2 * np.pi * doppler_shift * t)
    
    # Compense
    signal_compensated = controller.compensate_doppler(signal_doppler, doppler_shift, sample_rate)
    
    print(f"Signal original: {len(signal)} échantillons")
    print(f"Décalage Doppler appliqué: {doppler_shift} Hz")
    print(f"Signal compensé: {len(signal_compensated)} échantillons")
    print(f"Corrélation avant/après: {np.corrcoef(signal, signal_compensated)[0,1]:.3f}")
    
    # Test 5: Boucle d'adaptation complète
    print("\n--- Test 5: Boucle d'adaptation complète ---")
    controller = AdaptiveController(mode=AdaptiveMode.FULL_AUTO)
    
    # Simule 10 itérations avec conditions variables
    for i in range(10):
        # Simule une dégradation progressive
        noise_level = -110 + i * 5
        signal_rx = signal_tx.copy()
        num_errors = int(len(signal_rx) * i * 0.01)
        if num_errors > 0:
            error_idx = np.random.choice(len(signal_rx), num_errors, replace=False)
            signal_rx[error_idx] = 1 - signal_rx[error_idx]
        
        channel = controller.estimate_channel(signal_rx, signal_tx, -90, noise_level)
        params = controller.update(channel)
        
        print(f"It.{i+1}: SNR={channel.snr_db:.1f}dB BER={channel.ber:.2e} "
              f"→ Pwr={params.tx_power_dbm:.1f}dBm Rate={params.data_rate}bps SF={params.spreading_factor}")
    
    # Statistiques
    print("\n--- Statistiques d'adaptation ---")
    stats = controller.get_statistics()
    for key, value in stats.items():
        print(f"{key}: {value}")
    
    print("\n✓ Tests du contrôleur adaptatif réussis!")
