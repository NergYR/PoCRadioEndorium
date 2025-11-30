"""
Module de simulation d'effet Doppler pour communications radio mobiles
"""

import numpy as np
from typing import Tuple
from dataclasses import dataclass


@dataclass
class MobileScenario:
    """Scénario de mobilité"""
    velocity_ms: float  # Vitesse en m/s
    angle_deg: float    # Angle de déplacement par rapport à la ligne de vue (0° = vers l'émetteur)
    distance_m: float   # Distance initiale
    duration_s: float   # Durée de la simulation


class DopplerChannel:
    """Canal radio avec effet Doppler"""
    
    def __init__(self, carrier_freq_hz: float = 915e6):
        """
        Initialise le canal Doppler
        
        Args:
            carrier_freq_hz: Fréquence porteuse en Hz
        """
        self.carrier_freq = carrier_freq_hz
        self.speed_of_light = 3e8  # m/s
        
    def calculate_doppler_shift(self, velocity_ms: float, angle_deg: float) -> float:
        """
        Calcule le décalage Doppler
        
        Args:
            velocity_ms: Vitesse relative en m/s
            angle_deg: Angle de déplacement (0° = vers l'émetteur, 180° = s'éloigne)
            
        Returns:
            Décalage de fréquence en Hz
        """
        # Composante radiale de la vitesse
        angle_rad = np.deg2rad(angle_deg)
        radial_velocity = velocity_ms * np.cos(angle_rad)
        
        # Formule Doppler: Δf = (v/c) * f_carrier
        doppler_shift = (radial_velocity / self.speed_of_light) * self.carrier_freq
        
        return doppler_shift
    
    def calculate_doppler_spread(self, velocity_ms: float) -> float:
        """
        Calcule l'étalement Doppler (pour mouvement omnidirectionnel)
        
        Args:
            velocity_ms: Vitesse maximale
            
        Returns:
            Étalement Doppler en Hz
        """
        # Étalement Doppler = 2 * f_doppler_max
        f_doppler_max = (velocity_ms / self.speed_of_light) * self.carrier_freq
        return 2 * f_doppler_max
    
    def apply_doppler_to_signal(self, signal: np.ndarray, 
                                sample_rate: float,
                                doppler_shift_hz: float) -> np.ndarray:
        """
        Applique un décalage Doppler à un signal
        
        Args:
            signal: Signal complexe à décaler
            sample_rate: Taux d'échantillonnage
            doppler_shift_hz: Décalage en Hz
            
        Returns:
            Signal avec décalage Doppler
        """
        # Génère un vecteur temporel
        n_samples = len(signal)
        t = np.arange(n_samples) / sample_rate
        
        # Applique le décalage en fréquence via multiplication complexe
        phase_shift = 2 * np.pi * doppler_shift_hz * t
        doppler_signal = signal * np.exp(1j * phase_shift)
        
        return doppler_signal
    
    def simulate_rayleigh_fading(self, signal: np.ndarray,
                                 doppler_spread_hz: float,
                                 sample_rate: float) -> np.ndarray:
        """
        Simule un évanouissement de Rayleigh avec Doppler
        
        Args:
            signal: Signal d'entrée
            doppler_spread_hz: Étalement Doppler
            sample_rate: Taux d'échantillonnage
            
        Returns:
            Signal avec évanouissement de Rayleigh
        """
        n_samples = len(signal)
        
        # Génère deux processus gaussiens indépendants (parties I et Q)
        # Filtrés par un filtre Doppler (approximation via moyenne mobile)
        filter_length = max(1, int(sample_rate / doppler_spread_hz))
        
        i_component = np.random.randn(n_samples)
        q_component = np.random.randn(n_samples)
        
        # Filtre moyenneur simple (approximation du spectre Doppler)
        if filter_length > 1:
            i_filtered = np.convolve(i_component, np.ones(filter_length)/filter_length, mode='same')
            q_filtered = np.convolve(q_component, np.ones(filter_length)/filter_length, mode='same')
        else:
            i_filtered = i_component
            q_filtered = q_component
        
        # Canal de Rayleigh = I + jQ
        rayleigh_channel = (i_filtered + 1j * q_filtered) / np.sqrt(2)
        
        # Applique au signal (multiplication complexe si signal est complexe)
        if np.iscomplexobj(signal):
            faded_signal = signal * rayleigh_channel
        else:
            faded_signal = signal * np.abs(rayleigh_channel)
        
        return faded_signal
    
    def simulate_mobile_trajectory(self, scenario: MobileScenario, 
                                   sample_rate: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Simule une trajectoire mobile et calcule le Doppler instantané
        
        Args:
            scenario: Scénario de mobilité
            sample_rate: Taux d'échantillonnage
            
        Returns:
            (temps, doppler_shifts) - Vecteurs temporel et décalages Doppler
        """
        # Nombre d'échantillons
        n_samples = int(scenario.duration_s * sample_rate)
        t = np.linspace(0, scenario.duration_s, n_samples)
        
        # Position initiale
        x0 = scenario.distance_m * np.cos(np.deg2rad(scenario.angle_deg))
        y0 = scenario.distance_m * np.sin(np.deg2rad(scenario.angle_deg))
        
        # Trajectoire (mouvement rectiligne uniforme)
        vx = scenario.velocity_ms * np.cos(np.deg2rad(scenario.angle_deg))
        vy = scenario.velocity_ms * np.sin(np.deg2rad(scenario.angle_deg))
        
        x = x0 + vx * t
        y = y0 + vy * t
        
        # Distance à chaque instant
        distances = np.sqrt(x**2 + y**2)
        
        # Vitesse radiale (dérivée de la distance)
        radial_velocities = np.gradient(distances, t)
        
        # Décalage Doppler instantané
        doppler_shifts = (radial_velocities / self.speed_of_light) * self.carrier_freq
        
        return t, doppler_shifts, distances


def velocity_from_kmh(kmh: float) -> float:
    """Convertit km/h en m/s"""
    return kmh / 3.6


def velocity_to_kmh(ms: float) -> float:
    """Convertit m/s en km/h"""
    return ms * 3.6


# Scénarios typiques airsoft
AIRSOFT_SCENARIOS = {
    'stationnaire': MobileScenario(
        velocity_ms=0,
        angle_deg=0,
        distance_m=200,
        duration_s=10
    ),
    'marche': MobileScenario(
        velocity_ms=velocity_from_kmh(5),  # 5 km/h
        angle_deg=45,
        distance_m=200,
        duration_s=10
    ),
    'course': MobileScenario(
        velocity_ms=velocity_from_kmh(15),  # 15 km/h
        angle_deg=30,
        distance_m=200,
        duration_s=10
    ),
    'vehicule': MobileScenario(
        velocity_ms=velocity_from_kmh(50),  # 50 km/h
        angle_deg=0,
        distance_m=500,
        duration_s=10
    ),
    'approche_rapide': MobileScenario(
        velocity_ms=velocity_from_kmh(20),  # 20 km/h
        angle_deg=0,  # Directement vers l'émetteur
        distance_m=300,
        duration_s=10
    ),
    'eloignement': MobileScenario(
        velocity_ms=velocity_from_kmh(10),  # 10 km/h
        angle_deg=180,  # S'éloigne
        distance_m=200,
        duration_s=10
    )
}


if __name__ == "__main__":
    print("=== Test du module Doppler ===\n")
    
    doppler = DopplerChannel(carrier_freq_hz=915e6)
    
    # Test 1: Décalage Doppler pour différentes vitesses
    print("--- Décalage Doppler (angle 0° = vers émetteur) ---")
    velocities = [5, 10, 15, 20, 50, 100]  # km/h
    
    for v_kmh in velocities:
        v_ms = velocity_from_kmh(v_kmh)
        
        # Vers l'émetteur (0°)
        shift_towards = doppler.calculate_doppler_shift(v_ms, 0)
        
        # S'éloigne (180°)
        shift_away = doppler.calculate_doppler_shift(v_ms, 180)
        
        # Perpendiculaire (90°)
        shift_perp = doppler.calculate_doppler_shift(v_ms, 90)
        
        print(f"{v_kmh} km/h ({v_ms:.1f} m/s):")
        print(f"  Vers émetteur:  {shift_towards:+.2f} Hz")
        print(f"  S'éloigne:      {shift_away:+.2f} Hz")
        print(f"  Perpendiculaire: {shift_perp:+.2f} Hz")
    
    # Test 2: Étalement Doppler
    print("\n--- Étalement Doppler ---")
    for v_kmh in [5, 15, 50]:
        v_ms = velocity_from_kmh(v_kmh)
        spread = doppler.calculate_doppler_spread(v_ms)
        print(f"{v_kmh} km/h: Étalement = {spread:.2f} Hz")
    
    # Test 3: Scénarios airsoft
    print("\n--- Scénarios Airsoft ---")
    for name, scenario in AIRSOFT_SCENARIOS.items():
        v_kmh = velocity_to_kmh(scenario.velocity_ms)
        shift = doppler.calculate_doppler_shift(scenario.velocity_ms, scenario.angle_deg)
        spread = doppler.calculate_doppler_spread(scenario.velocity_ms)
        
        print(f"\n{name.upper()}:")
        print(f"  Vitesse: {v_kmh:.1f} km/h")
        print(f"  Angle: {scenario.angle_deg}°")
        print(f"  Distance: {scenario.distance_m} m")
        print(f"  Décalage Doppler: {shift:+.2f} Hz")
        print(f"  Étalement Doppler: {spread:.2f} Hz")
    
    # Test 4: Application sur un signal
    print("\n--- Application sur signal ---")
    sample_rate = 1e6
    signal_duration = 0.001  # 1 ms
    n_samples = int(sample_rate * signal_duration)
    
    # Signal test (sinusoïde)
    t = np.arange(n_samples) / sample_rate
    test_signal = np.exp(2j * np.pi * 1000 * t)  # 1 kHz
    
    # Applique Doppler pour course (15 km/h)
    v_ms = velocity_from_kmh(15)
    doppler_shift = doppler.calculate_doppler_shift(v_ms, 0)
    shifted_signal = doppler.apply_doppler_to_signal(test_signal, sample_rate, doppler_shift)
    
    print(f"Signal test: 1 kHz, durée {signal_duration*1000} ms")
    print(f"Doppler shift: {doppler_shift:+.2f} Hz")
    print(f"Échantillons: {len(shifted_signal)}")
    
    print("\n✓ Tests du module Doppler réussis!")
