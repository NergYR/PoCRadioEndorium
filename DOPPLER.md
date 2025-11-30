# Simulation de Mobilité - Effet Doppler

## Vue d'ensemble

Le système radio inclut désormais une simulation complète de l'effet Doppler pour modéliser les communications en situation de mobilité (joueurs en mouvement, véhicules, etc.).

## Module `doppler.py`

### Classe `DopplerChannel`

Gère la simulation de l'effet Doppler sur les signaux radio.

#### Méthodes principales

```python
# Calcul du décalage Doppler
doppler_shift = doppler.calculate_doppler_shift(velocity_ms, angle_deg)
# Formule: Δf = (v/c) × f_carrier × cos(θ)

# Calcul de l'étalement Doppler
doppler_spread = doppler.calculate_doppler_spread(velocity_ms)
# Formule: Spread = 2 × f_doppler_max

# Application du Doppler au signal
shifted_signal = doppler.apply_doppler_to_signal(signal, sample_rate, doppler_shift)

# Évanouissement de Rayleigh (canal multitrajet)
faded_signal = doppler.simulate_rayleigh_fading(signal, doppler_spread, sample_rate)
```

### Scénarios prédéfinis

Six scénarios airsoft sont disponibles via `AIRSOFT_SCENARIOS` :

| Scénario | Vitesse | Description |
|----------|---------|-------------|
| `stationnaire` | 0 km/h | Joueur immobile |
| `marche` | 5 km/h | Déplacement à pied |
| `course` | 15 km/h | Course rapide |
| `vehicule` | 50 km/h | Véhicule |
| `approche_rapide` | 20 km/h | Se rapproche de l'émetteur |
| `eloignement` | 10 km/h | S'éloigne de l'émetteur |

### Utilisation dans `simulator.py`

```python
from src.doppler import AIRSOFT_SCENARIOS, MobileScenario

# Configuration avec mobilité
config = RadioConfig(
    use_dsss=True,
    encryption_enabled=True,
    mobile_scenario=AIRSOFT_SCENARIOS['course'],
    enable_rayleigh_fading=True  # Active l'évanouissement
)

sim = RadioSimulator(config)

# Simulation avec Doppler à un instant t
result = sim.simulate_transmission(
    packet_size_bits=32, 
    distance_m=200,
    time_offset_s=5.0  # Position après 5 secondes
)
```

## Résultats de simulation

### Décalages Doppler typiques @ 915 MHz

| Vitesse | Décalage Doppler (0°) | Étalement Doppler |
|---------|----------------------|-------------------|
| 5 km/h (marche) | ±4.2 Hz | 8.5 Hz |
| 15 km/h (course) | ±12.7 Hz | 25.4 Hz |
| 50 km/h (véhicule) | ±42.4 Hz | 84.7 Hz |
| 100 km/h | ±84.7 Hz | 169.4 Hz |

**Note** : À 915 MHz, l'effet Doppler est relativement faible (quelques dizaines de Hz) même à haute vitesse, donc l'impact sur un système DSSS avec chip rate de 1 MHz est minimal.

### Performance BER

D'après les simulations :
- **BER moyen = 0** pour toutes les vitesses jusqu'à 100 km/h à courte distance (< 300m)
- Le SNR reste élevé (> 79 dB) dans tous les scénarios testés
- La portée n'est pas significativement affectée par la mobilité pour ce système

### Impact de l'angle

L'effet Doppler dépend de l'angle de déplacement :
- **0°** (vers l'émetteur) : Doppler positif maximal
- **90°** (perpendiculaire) : Doppler nul
- **180°** (s'éloigne) : Doppler négatif maximal

Formule : `Doppler = f_max × cos(θ)`

## Graphiques générés

### `mobility_comparison.png`
4 graphiques montrant l'évolution dans le temps pour différents scénarios :
- Distance vs temps
- BER vs temps
- SNR vs temps
- Décalages Doppler (graphique à barres)

### `velocity_impact.png`
2 graphiques d'analyse :
- BER en fonction de la vitesse (pour différentes distances)
- Décalage et étalement Doppler vs vitesse

## Exemple d'utilisation

```bash
# Exécuter la simulation de mobilité
python -m examples.mobility_sim
```

## Paramètres physiques

- **Fréquence porteuse** : 915 MHz (bande ISM)
- **Vitesse de la lumière** : 3×10⁸ m/s
- **Chip rate** : 1 MHz (DSSS)
- **Data rate** : 10 kbps

## Modèles implémentés

1. **Doppler classique** : Décalage de fréquence dû à la vitesse relative
2. **Étalement Doppler** : Dispersion spectrale pour mouvement omnidirectionnel
3. **Évanouissement de Rayleigh** : Canal multitrajet avec processus gaussiens filtrés (optionnel)

## Limitations et considérations

- L'effet Doppler à 915 MHz est **faible** comparé au chip rate (1 MHz)
- Un système DSSS avec spreading factor de 100 est **très robuste** au Doppler
- L'évanouissement de Rayleigh a un impact plus important que le Doppler pur
- Pour des vitesses > 100 km/h, il faudrait activer le mode Rayleigh pour plus de réalisme

## Formules de référence

### Décalage Doppler
```
f_d = (v/c) × f_c × cos(θ)
```
- `v` : vitesse relative (m/s)
- `c` : vitesse de la lumière (3×10⁸ m/s)
- `f_c` : fréquence porteuse (Hz)
- `θ` : angle entre direction de déplacement et ligne de vue

### Étalement Doppler
```
B_d = 2 × (v/c) × f_c
```

### Application au signal
```
s_doppler(t) = s(t) × exp(j2π × f_d × t)
```

## Prochaines améliorations possibles

- [ ] Canal de Rice (Rician fading) pour ligne de vue dominante
- [ ] Trajectoires courbes (pas seulement rectilignes)
- [ ] Modèle de mobilité stochastique (mouvement aléatoire)
- [ ] Compensation Doppler au récepteur
- [ ] Analyse de la cohérence temporelle du canal
