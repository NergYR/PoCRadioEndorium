# Intégration GNURadio - Système Radio Airsoft

## 📡 Vue d'ensemble

Ce projet inclut une intégration complète avec GNURadio pour visualiser et traiter les signaux radio DSSS chiffrés en temps réel.

## 🚀 Démarrage rapide

### 1. Générer les signaux

```powershell
C:/Users/energ/Desktop/Code/PoCRadio/.venv/Scripts/python.exe export_gnuradio.py
```

### 2. Ouvrir le flowgraph dans GNURadio Companion

```bash
gnuradio-companion gnuradio_airsoft_receiver.grc
```

### 3. Exécuter la simulation

Appuyez sur **F5** ou cliquez sur **Execute** dans GNURadio Companion.

## 📁 Fichiers générés

| Fichier | Description | Format |
|---------|-------------|---------|
| `gnuradio_dsss_signal.bin` | Signal DSSS propre (sans bruit) | Complex Float32 (IQ) |
| `gnuradio_dsss_noisy.bin` | Signal avec bruit (SNR=15dB) | Complex Float32 (IQ) |
| `gnuradio_pn_sequence.bin` | Séquence PN pour désétalement | Float32 |
| `gnuradio_metadata.csv` | Paramètres du signal | CSV |
| `gnuradio_airsoft_receiver.grc` | Flowgraph GNURadio | XML/YAML |

## 🔧 Architecture du flowgraph

```
┌─────────────────┐
│  File Source    │  Signal DSSS bruité
│  (Complex)      │  
└────────┬────────┘
         │
         ▼
┌─────────────────┐     ┌──────────────┐
│ Complex to Real │────▶│  Multiply    │  Désétalement
└─────────────────┘     │  (×PN)       │
                        └──────┬───────┘
         ┌──────────────────────┘
         │
         ▼
┌─────────────────┐
│   Integrate     │  Intègre sur spreading_factor (100)
│   (Decim=100)   │
└────────┬────────┘
         │
         ├──────────▶ Constellation Plot
         │
         ├──────────▶ Time Sink (visualisation temporelle)
         │
         ▼
┌─────────────────┐
│   Threshold     │  Décision binaire (seuil à 0)
│   (FF)          │
└─────────────────┘
```

## 📊 Visualisations disponibles

1. **Signal temporel** : Signal reçu vs signal désétalé
2. **Constellation** : Diagramme de constellation après désétalement
3. **FFT** : Spectre de fréquence (ajoutez un FFT Sink si besoin)

## ⚙️ Paramètres clés

- **Sample Rate** : 1 MHz (1 000 000 Hz)
- **Spreading Factor** : 100
- **Data Rate** : 10 kbps
- **Fréquence centrale** : 915 MHz
- **Modulation** : DSSS-BPSK

## 🎯 Utilisation avec du matériel réel

### Option 1 : RTL-SDR (Réception uniquement)

Remplacez `File Source` par `RTL-SDR Source` :

```
┌─────────────────┐
│  RTL-SDR Source │
│  Freq: 915 MHz  │
│  Rate: 1 MHz    │
│  Gain: 30 dB    │
└─────────────────┘
```

### Option 2 : HackRF / USRP (Émission + Réception)

**Émetteur** :
```python
# Générer signal à transmettre
python export_gnuradio.py

# Dans GRC, ajouter:
# File Source → HackRF/USRP Sink
```

**Récepteur** :
```
HackRF/USRP Source → [flowgraph existant]
```

### Configuration matérielle recommandée

| Matériel | Rôle | Prix approx. |
|----------|------|--------------|
| HackRF One | TX/RX | ~300€ |
| USRP B200 | TX/RX (meilleur) | ~800€ |
| RTL-SDR | RX uniquement | ~30€ |

## 🔐 Chiffrement

**Note importante** : Le chiffrement AES-CTR+HMAC est appliqué **avant** la modulation DSSS dans la simulation Python. Pour un système réel avec GNURadio :

1. **Option A** : Implémenter AES en Python (GNU Radio embedded Python block)
2. **Option B** : Utiliser un OOT module (Out-Of-Tree) pour crypto
3. **Option C** : Chiffrer en amont et transmettre le ciphertext

### Exemple avec Python Block

```python
import numpy as np
from Crypto.Cipher import AES
from gnuradio import gr

class aes_encryptor(gr.sync_block):
    def __init__(self, key):
        gr.sync_block.__init__(
            self,
            name="AES Encryptor",
            in_sig=[np.uint8],
            out_sig=[np.uint8]
        )
        self.cipher = AES.new(key, AES.MODE_CTR)
    
    def work(self, input_items, output_items):
        # Chiffre les données
        output_items[0][:] = self.cipher.encrypt(input_items[0].tobytes())
        return len(output_items[0])
```

## 📈 Tests et validation

### Test 1 : Vérifier le désétalement

```bash
# Comparer BER avant/après désétalement
python -c "
from src.modulation import DSSSModulator
import numpy as np

mod = DSSSModulator()
data = np.random.randint(0, 2, 100)
spread = mod.spread(data)
despread = mod.despread(spread)
print(f'BER: {np.sum(data != despread) / len(data)}')
"
```

### Test 2 : Spectre de fréquence

Dans GNURadio, ajoutez un **QT GUI Frequency Sink** après le File Source.

### Test 3 : Portée simulée

Modifiez le SNR dans `export_gnuradio.py` :

```python
# Test avec différents SNR
for snr in [5, 10, 15, 20]:
    noisy = add_awgn(signal, snr_db=snr)
    # Export et test dans GNURadio
```

## 🛠️ Dépannage

### Problème : "File not found"

Assurez-vous que les fichiers `.bin` sont dans le même répertoire que le `.grc`.

### Problème : Signal trop faible

Ajustez le gain dans GNURadio :
- RTL-SDR : 30-40 dB
- HackRF : 14-30 dB (TX), 40 dB (RX)

### Problème : Constellation floue

- Vérifiez l'alignement de la séquence PN
- Ajoutez une synchronisation temporelle (Polyphase Clock Sync)
- Ajustez le seuil de décision

## 🚀 Améliorations futures

- [ ] Ajout de FEC (Forward Error Correction)
- [ ] Synchronisation automatique (Costas Loop)
- [ ] Égaliseur adaptatif
- [ ] Support multi-utilisateurs (CDMA)
- [ ] Interface de contrôle (start/stop via socket)

## 📚 Ressources

- [GNURadio Tutorials](https://wiki.gnuradio.org/index.php/Tutorials)
- [GNURadio Flowgraph](https://wiki.gnuradio.org/index.php/Flowgraph_Python_Code)
- [DSSS Theory](https://en.wikipedia.org/wiki/Direct-sequence_spread_spectrum)

## 💡 Conseil pour airsoft réel

Pour une utilisation terrain :

1. **Licence radio** : Vérifiez la réglementation de votre pays (ISM 433/868/915 MHz)
2. **Puissance** : Limitez à 14 dBm (25 mW) en Europe pour ISM
3. **Antennes** : Dipôle 1/4 onde (~8cm à 915 MHz)
4. **Batterie** : HackRF consomme ~500mA @ 5V
5. **Boîtier** : Protégez le matériel (waterproof recommandé)

---

**Projet** : PoCRadio - Système radio airsoft alternatif LoRa
**Version** : 0.1.0
**Dernière MAJ** : 30 novembre 2025
