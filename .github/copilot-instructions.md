# Instructions Copilot - Simulation Système Radio Airsoft

## Contexte du projet
Projet de simulation Python pour un système radio alternatif au LoRa destiné aux parties d'airsoft (intérieur et extérieur).

## Caractéristiques techniques
- Étalement de spectre (DSSS/FHSS)
- Longue portée (jusqu'à 1,8 km en extérieur)
- Chiffrement AES-256 des communications
- Simulation de propagation indoor/outdoor
- Analyse de performance (BER, PER, SNR)
- Visualisation graphique

## Structure du projet
```
src/
├── modulation.py    # DSSS et FHSS
├── crypto.py        # Chiffrement AES-256
├── propagation.py   # Modèles de propagation
└── simulator.py     # Simulateur complet

examples/
├── compare_modulations.py
├── indoor_vs_outdoor.py
└── plot_range.py
```

## État d'avancement
- [x] Création du fichier copilot-instructions.md
- [x] Configuration du projet Python
- [x] Structure du projet créée
- [x] Code de simulation implémenté
- [x] Documentation complétée

## Utilisation
1. Installer les dépendances : `pip install -r requirements.txt`
2. Tester un module : `python src/modulation.py`
3. Exécuter une simulation : `python examples/plot_range.py`
