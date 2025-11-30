"""
Module de calcul de propagation radio et de portée
"""

import numpy as np
from enum import Enum
from typing import Tuple


class Environment(Enum):
    """Types d'environnement pour la propagation"""
    OUTDOOR_OPEN = "outdoor_open"  # Extérieur dégagé
    OUTDOOR_SUBURBAN = "outdoor_suburban"  # Extérieur suburbain
    OUTDOOR_URBAN = "outdoor_urban"  # Extérieur urbain
    INDOOR_OFFICE = "indoor_office"  # Intérieur bureau
    INDOOR_BUILDING = "indoor_building"  # Intérieur bâtiment dense


class PropagationModel:
    """Modèles de propagation radio"""
    
    def __init__(self, frequency_mhz: float = 915.0, tx_power_dbm: float = 20.0):
        """
        Initialise le modèle de propagation
        
        Args:
            frequency_mhz: Fréquence en MHz
            tx_power_dbm: Puissance d'émission en dBm
        """
        self.frequency_mhz = frequency_mhz
        self.tx_power_dbm = tx_power_dbm
        self.frequency_hz = frequency_mhz * 1e6
        
    def free_space_path_loss(self, distance_m: float) -> float:
        """
        Calcule l'affaiblissement en espace libre (FSPL)
        
        Args:
            distance_m: Distance en mètres
            
        Returns:
            Perte en dB
        """
        if distance_m <= 0:
            return 0
        
        # FSPL = 20*log10(d) + 20*log10(f) + 32.45
        # où d est en km et f en MHz
        fspl = 20 * np.log10(distance_m / 1000) + 20 * np.log10(self.frequency_mhz) + 32.45
        return fspl
    
    def two_ray_ground_reflection(self, distance_m: float, ht: float = 1.5, hr: float = 1.5) -> float:
        """
        Modèle à deux rayons avec réflexion au sol
        
        Args:
            distance_m: Distance en mètres
            ht: Hauteur de l'antenne émettrice (m)
            hr: Hauteur de l'antenne réceptrice (m)
            
        Returns:
            Perte en dB
        """
        if distance_m <= 0:
            return 0
        
        # Formule simplifiée pour grande distance
        wavelength = 3e8 / self.frequency_hz
        
        if distance_m < (4 * ht * hr / wavelength):
            # Zone de Fresnel - utilise FSPL
            return self.free_space_path_loss(distance_m)
        else:
            # Zone lointaine
            path_loss = 40 * np.log10(distance_m) - 20 * np.log10(ht) - 20 * np.log10(hr)
            return path_loss
    
    def log_distance_path_loss(self, distance_m: float, environment: Environment) -> float:
        """
        Modèle log-distance avec exposant d'affaiblissement selon l'environnement
        
        Args:
            distance_m: Distance en mètres
            environment: Type d'environnement
            
        Returns:
            Perte en dB
        """
        if distance_m <= 0:
            return 0
        
        # Distance de référence
        d0 = 1.0  # 1 mètre
        
        # Exposant d'affaiblissement selon l'environnement
        path_loss_exponents = {
            Environment.OUTDOOR_OPEN: 2.0,
            Environment.OUTDOOR_SUBURBAN: 3.0,
            Environment.OUTDOOR_URBAN: 3.5,
            Environment.INDOOR_OFFICE: 3.0,
            Environment.INDOOR_BUILDING: 4.0,
        }
        
        n = path_loss_exponents.get(environment, 2.0)
        
        # Perte à la distance de référence
        pl_d0 = self.free_space_path_loss(d0)
        
        # PL(d) = PL(d0) + 10*n*log10(d/d0)
        path_loss = pl_d0 + 10 * n * np.log10(distance_m / d0)
        
        return path_loss
    
    def received_power(self, distance_m: float, environment: Environment, 
                       tx_gain_dbi: float = 2.0, rx_gain_dbi: float = 2.0,
                       additional_loss_db: float = 0.0) -> float:
        """
        Calcule la puissance reçue
        
        Args:
            distance_m: Distance en mètres
            environment: Type d'environnement
            tx_gain_dbi: Gain de l'antenne émettrice en dBi
            rx_gain_dbi: Gain de l'antenne réceptrice en dBi
            additional_loss_db: Pertes supplémentaires (câbles, connecteurs, etc.)
            
        Returns:
            Puissance reçue en dBm
        """
        # Perte de propagation
        path_loss = self.log_distance_path_loss(distance_m, environment)
        
        # Bilan de liaison: Pr = Pt + Gt + Gr - PL - Losses
        rx_power = self.tx_power_dbm + tx_gain_dbi + rx_gain_dbi - path_loss - additional_loss_db
        
        return rx_power
    
    def link_budget(self, distance_m: float, environment: Environment,
                    rx_sensitivity_dbm: float = -110.0,
                    tx_gain_dbi: float = 2.0, rx_gain_dbi: float = 2.0,
                    fade_margin_db: float = 10.0) -> Tuple[bool, float]:
        """
        Calcule le bilan de liaison et détermine si la liaison est possible
        
        Args:
            distance_m: Distance en mètres
            environment: Type d'environnement
            rx_sensitivity_dbm: Sensibilité du récepteur en dBm
            tx_gain_dbi: Gain antenne émettrice
            rx_gain_dbi: Gain antenne réceptrice
            fade_margin_db: Marge d'évanouissement requise
            
        Returns:
            (liaison_possible, marge_disponible_db)
        """
        rx_power = self.received_power(distance_m, environment, tx_gain_dbi, rx_gain_dbi)
        
        # Marge disponible
        margin = rx_power - rx_sensitivity_dbm
        
        # La liaison est possible si la marge dépasse le fade margin requis
        link_ok = margin >= fade_margin_db
        
        return link_ok, margin
    
    def max_range(self, environment: Environment, 
                  rx_sensitivity_dbm: float = -110.0,
                  tx_gain_dbi: float = 2.0, rx_gain_dbi: float = 2.0,
                  fade_margin_db: float = 10.0,
                  step_m: float = 10.0, max_distance_m: float = 10000.0) -> float:
        """
        Calcule la portée maximale
        
        Args:
            environment: Type d'environnement
            rx_sensitivity_dbm: Sensibilité du récepteur
            tx_gain_dbi: Gain antenne émettrice
            rx_gain_dbi: Gain antenne réceptrice
            fade_margin_db: Marge d'évanouissement
            step_m: Pas de recherche en mètres
            max_distance_m: Distance maximale à tester
            
        Returns:
            Portée maximale en mètres
        """
        distance = 0
        while distance < max_distance_m:
            link_ok, margin = self.link_budget(
                distance, environment, rx_sensitivity_dbm,
                tx_gain_dbi, rx_gain_dbi, fade_margin_db
            )
            
            if not link_ok:
                return distance - step_m
            
            distance += step_m
        
        return max_distance_m


def calculate_fresnel_zone(distance_m: float, frequency_mhz: float, zone_number: int = 1) -> float:
    """
    Calcule le rayon de la zone de Fresnel
    
    Args:
        distance_m: Distance totale entre émetteur et récepteur
        frequency_mhz: Fréquence en MHz
        zone_number: Numéro de la zone de Fresnel (1, 2, 3...)
        
    Returns:
        Rayon de la zone en mètres
    """
    c = 3e8  # Vitesse de la lumière
    wavelength = c / (frequency_mhz * 1e6)
    
    # Rayon au point milieu
    radius = np.sqrt(zone_number * wavelength * distance_m / 4)
    
    return radius


if __name__ == "__main__":
    print("=== Simulation de propagation radio ===")
    
    # Configuration du système
    prop = PropagationModel(frequency_mhz=915.0, tx_power_dbm=20.0)
    
    # Test pour différents environnements
    environments = [
        (Environment.OUTDOOR_OPEN, "Extérieur dégagé (airsoft terrain ouvert)"),
        (Environment.OUTDOOR_SUBURBAN, "Extérieur suburbain"),
        (Environment.OUTDOOR_URBAN, "Extérieur urbain (airsoft CQB ville)"),
        (Environment.INDOOR_BUILDING, "Intérieur bâtiment (airsoft CQB indoor)"),
    ]
    
    print(f"\nConfiguration:")
    print(f"  Fréquence: {prop.frequency_mhz} MHz")
    print(f"  Puissance émission: {prop.tx_power_dbm} dBm")
    print(f"  Sensibilité récepteur: -110 dBm")
    print(f"  Gains antennes: 2 dBi (Tx/Rx)")
    
    for env, name in environments:
        print(f"\n--- {name} ---")
        
        # Portée maximale
        max_range_m = prop.max_range(env, rx_sensitivity_dbm=-110.0)
        print(f"Portée maximale: {max_range_m:.0f} m")
        
        # Test à différentes distances
        test_distances = [50, 100, 200, 500]
        for dist in test_distances:
            if dist <= max_range_m:
                link_ok, margin = prop.link_budget(dist, env)
                rx_pwr = prop.received_power(dist, env)
                print(f"  {dist}m: Rx={rx_pwr:.1f} dBm, Marge={margin:.1f} dB, Liaison={'OK' if link_ok else 'KO'}")
    
    # Zone de Fresnel
    print(f"\n--- Zone de Fresnel ---")
    for dist in [100, 500, 1000]:
        radius = calculate_fresnel_zone(dist, prop.frequency_mhz)
        print(f"Distance {dist}m: rayon 1ère zone = {radius:.2f} m")
