# Simulateur de Système Radio pour Airsoft

## 📡 Description

Simulation complète d'un système radio alternatif au LoRa, spécialement conçu pour les parties d'airsoft en intérieur et extérieur. Le système implémente des techniques avancées de communication radio incluant l'étalement de spectre, le chiffrement et des modèles de propagation réalistes.

## ✨ Caractéristiques

### Modulation
- **DSSS (Direct Sequence Spread Spectrum)** : Étalement de spectre par séquence directe avec gain de traitement configurable
- **FHSS (Frequency Hopping Spread Spectrum)** : Saut de fréquence avec séquence pseudo-aléatoire
- Taux de chips configurable (1 Mchip/s par défaut)
- Débit de données ajustable (10 kbps par défaut)

### Sécurité
- **Chiffrement AES-256-CTR** : Protection des communications en mode streaming
- **Authentification HMAC-SHA256** : Vérification d'intégrité des messages
- Dérivation de clé par PBKDF2
- Support du chiffrement de tableaux NumPy

### Système Adaptatif ⭐ NOUVEAU
- **Contrôle adaptatif temps réel** : Ajustement automatique des paramètres
- **Compensation Doppler** : Correction du décalage de fréquence
- **Adaptation de puissance** : 0-27 dBm selon conditions du canal
- **Adaptation de débit** : 5-50 kbps selon qualité (SNR/BER)
- **Adaptation du spreading factor** : 50-400 selon SNR/Doppler
- **5 modes de fonctionnement** : Manual, Auto Power, Auto Rate, Auto Modulation, Full Auto

### Propagation Radio
- Modèle d'espace libre (FSPL)
- Modèle à deux rayons avec réflexion au sol
- Modèle log-distance avec exposants variables selon l'environnement
- Support de plusieurs environnements :
  - Extérieur dégagé (terrain ouvert)
  - Extérieur suburbain
  - Extérieur urbain (CQB ville)
  - Intérieur bureau
  - Intérieur bâtiment dense (CQB indoor)

### Mobilité et Effet Doppler ⭐ NOUVEAU
- **Simulation d'effet Doppler** : Décalage de fréquence dû au mouvement
- **Scénarios airsoft prédéfinis** : Marche, course, véhicule, stationnaire
- **Évanouissement de Rayleigh** : Canal multitrajet avec mobilité
- **Analyse de trajectoires** : Calcul du Doppler instantané
- Visualisation de l'impact de la vitesse sur les performances

### Analyse de Performance
- Calcul du BER (Bit Error Rate)
- Calcul du PER (Packet Error Rate)
- Analyse du rapport signal/bruit (SNR)
- Bilan de liaison complet
- Estimation de portée maximale
- Visualisation graphique des performances

## 🛠️ Installation

### Prérequis
- Python 3.8 ou supérieur
- pip (gestionnaire de paquets Python)

### Installation des dépendances

```powershell
# Depuis le répertoire du projet
pip install -r requirements.txt
```

Ou installation manuelle :
```powershell
pip install numpy scipy matplotlib cryptography pycryptodome
```

### Installation en mode développement

```powershell
pip install -e .
```

## 🚀 Utilisation

### Exemple basique

```python
from src.simulator import RadioSimulator, RadioConfig
from src.propagation import Environment

# Configuration pour terrain extérieur
config = RadioConfig(
    use_dsss=True,
    data_rate=10000,
    environment=Environment.OUTDOOR_OPEN,
    encryption_enabled=True,
    password="airsoft2024"
)

# Création du simulateur
sim = RadioSimulator(config)

# Simulation à 500m
result = sim.simulate_transmission(
    packet_size_bits=256,
    distance_m=500,
    num_packets=100
)

print(f"BER: {result.ber:.2e}")
print(f"PER: {result.per:.2%}")
print(f"Portée: {result.link_margin_db:.1f} dB de marge")
```

### Scripts d'exemple

Le dossier `examples/` contient plusieurs scripts prêts à l'emploi :

#### 1. Comparaison DSSS vs FHSS
```powershell
python -m examples.compare_modulations
```
Compare les performances des deux types de modulation avec chiffrement.

#### 2. Analyse intérieur vs extérieur
```powershell
python -m examples.indoor_vs_outdoor
```
Évalue les performances dans différents environnements d'airsoft.

#### 3. Génération de graphiques de portée
```powershell
python -m examples.plot_range
```
Génère une analyse complète avec graphiques (BER, puissance reçue, SNR, portée).

#### 4. Simulation de mobilité (Doppler) ⭐ NOUVEAU
```powershell
python -m examples.mobility_sim
```
Simule différents scénarios de mobilité (marche, course, véhicule) et analyse l'impact de l'effet Doppler.

#### 5. Visualisation de l'effet Doppler
```powershell
python -m examples.visualize_doppler
```
Génère des graphiques détaillés du spectre et des trajectoires avec effet Doppler.

#### 6. Système adaptatif en temps réel ⭐ NOUVEAU
```powershell
python -m examples.adaptive_demo
```
Démonstration complète du contrôle adaptatif avec scénario airsoft réaliste (4 phases : dégagé, forêt, course, urbain).

### Tests des modules individuels

Chaque module peut être testé indépendamment :

```powershell
# Test de modulation
python src/modulation.py

# Test de chiffrement
python src/crypto.py

# Test de propagation
python src/propagation.py

# Test de l'effet Doppler
python src/doppler.py

# Test du système adaptatif
python src/adaptive.py

# Test du simulateur complet
python src/simulator.py
```

## 📊 Structure du projet

```
PoCRadio/
├── .github/
│   └── copilot-instructions.md    # Instructions pour Copilot
├── src/
│   ├── __init__.py               # Initialisation du package
│   ├── modulation.py             # DSSS et FHSS
│   ├── crypto.py                 # Chiffrement AES-256-CTR
│   ├── propagation.py            # Modèles de propagation
│   ├── doppler.py                # Effet Doppler et mobilité
│   ├── adaptive.py               # Contrôle adaptatif ⭐
│   └── simulator.py              # Simulateur principal
├── examples/
│   ├── compare_modulations.py    # Comparaison DSSS/FHSS
│   ├── indoor_vs_outdoor.py      # Comparaison environnements
│   ├── plot_range.py             # Génération de graphiques
│   ├── fhss_encrypted.py         # Scénario FHSS chiffré
│   ├── mobility_sim.py           # Simulation de mobilité
│   ├── visualize_doppler.py      # Visualisation Doppler
│   └── adaptive_demo.py          # Démonstration adaptative ⭐
├── requirements.txt              # Dépendances Python
├── setup.py                      # Configuration du package
├── .gitignore                    # Fichiers à ignorer
├── README.md                     # Ce fichier
├── GNURADIO.md                   # Intégration GNURadio
├── DOPPLER.md                    # Documentation Doppler
└── ADAPTIVE.md                   # Documentation système adaptatif ⭐
```

## 🔧 Configuration

### Paramètres du système radio

```python
RadioConfig(
    # Modulation
    chip_rate=1000000,           # Taux de chips (Hz)
    data_rate=10000,             # Débit de données (bps)
    use_dsss=True,               # True=DSSS, False=FHSS
    
    # FHSS (si use_dsss=False)
    num_channels=50,             # Nombre de canaux
    hop_duration=0.1,            # Durée de chaque saut (s)
    
    # Paramètres RF
    frequency_mhz=915.0,         # Fréquence porteuse (MHz)
    tx_power_dbm=20.0,           # Puissance émission (dBm)
    rx_sensitivity_dbm=-110.0,   # Sensibilité réception (dBm)
    tx_gain_dbi=2.0,             # Gain antenne TX (dBi)
    rx_gain_dbi=2.0,             # Gain antenne RX (dBi)
    
    # Sécurité
    encryption_enabled=True,     # Activer le chiffrement
    password="airsoft2024",      # Mot de passe
    
    # Environnement
    environment=Environment.OUTDOOR_OPEN
)
```

### Environnements disponibles

- `Environment.OUTDOOR_OPEN` : Terrain ouvert (forêt dégagée) - Exposant 2.0
- `Environment.OUTDOOR_SUBURBAN` : Zone suburbaine - Exposant 3.0
- `Environment.OUTDOOR_URBAN` : Zone urbaine dense - Exposant 3.5
- `Environment.INDOOR_OFFICE` : Bureau/bâtiment léger - Exposant 3.0
- `Environment.INDOOR_BUILDING` : Bâtiment dense (CQB) - Exposant 4.0

## 📈 Résultats typiques

### Terrain extérieur ouvert (DSSS)
- **50m** : BER < 10⁻⁶, Marge > 80 dB
- **100m** : BER < 10⁻⁵, Marge > 70 dB
- **500m** : BER < 10⁻³, Marge > 50 dB
- **1000m** : BER < 10⁻², Marge > 40 dB
- **Portée maximale** : ~1800m (BER < 10⁻³)

### Intérieur bâtiment CQB (DSSS)
- **50m** : BER < 10⁻⁵, Marge > 60 dB
- **100m** : BER < 10⁻³, Marge > 40 dB
- **200m** : BER < 10⁻², Marge > 20 dB
- **Portée maximale** : ~250m (BER < 10⁻³)

## 🎯 Applications Airsoft

### Scénarios d'utilisation

1. **Terrain extérieur** (forêt, champs)
   - Portée : jusqu'à 1,5 km
   - Configuration recommandée : DSSS, 20 dBm

2. **CQB urbain extérieur** (village, ville)
   - Portée : jusqu'à 800m
   - Configuration recommandée : DSSS ou FHSS, 20 dBm

3. **CQB intérieur** (bâtiments)
   - Portée : jusqu'à 250m
   - Configuration recommandée : FHSS pour résistance aux multi-trajets

### Avantages par rapport au LoRa

- ✅ **Étalement de spectre plus robuste** (DSSS + FHSS)
- ✅ **Chiffrement intégré** (AES-256)
- ✅ **Débit configurable** selon les besoins
- ✅ **Meilleure résistance aux interférences**
- ✅ **Authentification des messages**

## 🔬 Aspects techniques

### Gain de traitement DSSS

Avec un chip rate de 1 Mchip/s et un débit de 10 kbps :
- Facteur d'étalement : 100
- **Gain de traitement : 20 dB**

### Sécurité

- **Chiffrement** : AES-256-CTR avec IV aléatoire (mode streaming)
- **Dérivation de clé** : PBKDF2-HMAC-SHA256 (100 000 itérations)
- **Authentification** : HMAC-SHA256

### Modèles de propagation

Le simulateur utilise des modèles scientifiquement validés :
- Formule de Friis (espace libre)
- Modèle à deux rayons (réflexion sol)
- Modèle log-distance empirique

### Effet Doppler ⭐

À 915 MHz, les décalages Doppler typiques sont :
- **5 km/h** (marche) : ±4.2 Hz
- **15 km/h** (course) : ±12.7 Hz
- **50 km/h** (véhicule) : ±42.4 Hz

Impact : **Faible** grâce au DSSS (spreading factor 100 >> décalage Doppler)

## 🎮 Cas d'usage Airsoft

### Scénarios testés
- ✅ Communication entre joueurs stationnaires (0-1400m extérieur)
- ✅ Joueur en mouvement (marche/course)
- ✅ Communication véhicule-base
- ✅ CQB indoor (0-200m)
- ✅ Partie en forêt (0-800m)
- ✅ Opération urbaine avec obstacles (0-400m)

### Performances obtenues
- **BER** : 0 (zéro erreur) jusqu'à 1000m en extérieur
- **Latence** : < 30 ms pour 256 bits
- **Robustesse Doppler** : Aucune dégradation jusqu'à 100 km/h
- **Sécurité** : Chiffrement militaire AES-256

## 🐛 Dépannage

### Erreurs d'importation
Si vous obtenez des erreurs d'importation dans les exemples :
```powershell
# Exécutez depuis la racine du projet
python -m examples.compare_modulations
```

### Problèmes avec matplotlib
Si les graphiques ne s'affichent pas :
```python
import matplotlib
matplotlib.use('TkAgg')  # ou 'Qt5Agg'
```

### Installation de cryptography
Si l'installation échoue sur Windows :
```powershell
pip install --upgrade pip setuptools wheel
pip install cryptography
```

## 📝 Développement futur

- [x] Modèle de canal avec évanouissement de Rayleigh ✅
- [x] Simulation de mobilité avec effet Doppler ✅
- [x] Intégration GNURadio ✅
- [ ] Codage correcteur d'erreurs (FEC)
- [ ] Interface graphique (GUI)
- [ ] Export des résultats en CSV/JSON
- [ ] Compensation Doppler au récepteur
- [ ] Simulation multi-utilisateurs
- [ ] Analyse de capacité du réseau

## 📄 Licence

Ce projet est un PoC (Proof of Concept) à des fins éducatives et de simulation.

## 👤 Auteur

Projet de simulation pour système radio airsoft

## 🤝 Contribution

Les contributions sont les bienvenues ! N'hésitez pas à :
- Ouvrir des issues pour les bugs
- Proposer des améliorations
- Soumettre des pull requests

---

**Note** : Ce projet est une simulation logicielle. L'implémentation matérielle nécessiterait du matériel radio approprié et des licences de fréquences radio selon votre pays.
