"""
Module de chiffrement pour les communications radio
"""

from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import os
import numpy as np


class RadioCrypto:
    """Chiffrement AES-256 pour les communications radio (AES-CTR + HMAC-SHA256)"""
    
    def __init__(self, password: str = None):
        """
        Initialise le système de chiffrement
        
        Args:
            password: Mot de passe pour la dérivation de clé (optionnel)
        """
        if password:
            self.key = self._derive_key(password)
        else:
            # Génère une clé aléatoire de 256 bits
            self.key = os.urandom(32)
    
    def _derive_key(self, password: str, salt: bytes = None) -> bytes:
        """
        Dérive une clé de 256 bits à partir d'un mot de passe
        
        Args:
            password: Mot de passe
            salt: Sel pour la dérivation (généré si None)
            
        Returns:
            Clé de 256 bits
        """
        if salt is None:
            salt = b'airsoftradio2024'  # Salt fixe pour partager entre émetteur/récepteur
        
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
            backend=default_backend()
        )
        
        return kdf.derive(password.encode())
    
    def encrypt(self, plaintext: bytes) -> tuple:
        """
        Chiffre les données avec AES-256-CTR et ajoute un HMAC-SHA256.
        
        Args:
            plaintext: Données à chiffrer
            
        Returns:
            (iv, ciphertext, mac)
        """
        iv = os.urandom(16)
        cipher = Cipher(
            algorithms.AES(self.key),
            modes.CTR(iv),
            backend=default_backend()
        )
        encryptor = cipher.encryptor()
        ciphertext = encryptor.update(plaintext) + encryptor.finalize()

        # HMAC pour intégrité
        from cryptography.hazmat.primitives import hmac
        h = hmac.HMAC(self.key, hashes.SHA256(), backend=default_backend())
        h.update(iv + ciphertext)
        mac = h.finalize()

        return iv, ciphertext, mac
    
    def decrypt(self, iv: bytes, ciphertext: bytes, mac: bytes) -> bytes:
        """
        Déchiffre les données (AES-CTR) et vérifie HMAC.
        """
        # Vérifie HMAC
        from cryptography.hazmat.primitives import hmac
        h = hmac.HMAC(self.key, hashes.SHA256(), backend=default_backend())
        h.update(iv + ciphertext)
        h.verify(mac)

        cipher = Cipher(
            algorithms.AES(self.key),
            modes.CTR(iv),
            backend=default_backend()
        )
        decryptor = cipher.decryptor()
        plaintext = decryptor.update(ciphertext) + decryptor.finalize()
        return plaintext
    
    def encrypt_array(self, data: np.ndarray) -> tuple:
        """
        Chiffre un tableau NumPy
        
        Args:
            data: Tableau à chiffrer
            
        Returns:
            (iv, ciphertext, shape, dtype) pour la reconstruction
        """
        # Sauvegarde la forme et le type
        shape = data.shape
        dtype = str(data.dtype)
        
        # Convertit en bytes
        data_bytes = data.tobytes()
        
        # Chiffre
        iv, ciphertext, mac = self.encrypt(data_bytes)
        return iv, ciphertext, mac, shape, dtype
    
    def decrypt_array(self, iv: bytes, ciphertext: bytes, mac: bytes, shape: tuple, dtype: str) -> np.ndarray:
        """
        Déchiffre un tableau NumPy
        
        Args:
            iv: Vecteur d'initialisation
            ciphertext: Données chiffrées
            shape: Forme du tableau original
            dtype: Type de données
            
        Returns:
            Tableau déchiffré
        """
        # Déchiffre
        data_bytes = self.decrypt(iv, ciphertext, mac)
        
        # Reconstruit le tableau
        data = np.frombuffer(data_bytes, dtype=dtype)
        return data.reshape(shape)


class MessageAuthenticator:
    """Authentification des messages pour détecter les altérations"""
    
    def __init__(self, key: bytes = None):
        """
        Initialise l'authentificateur
        
        Args:
            key: Clé pour HMAC (générée si None)
        """
        self.key = key if key else os.urandom(32)
    
    def compute_mac(self, message: bytes) -> bytes:
        """
        Calcule le MAC (Message Authentication Code) d'un message
        
        Args:
            message: Message à authentifier
            
        Returns:
            MAC du message
        """
        from cryptography.hazmat.primitives import hmac
        
        h = hmac.HMAC(self.key, hashes.SHA256(), backend=default_backend())
        h.update(message)
        return h.finalize()
    
    def verify_mac(self, message: bytes, mac: bytes) -> bool:
        """
        Vérifie le MAC d'un message
        
        Args:
            message: Message à vérifier
            mac: MAC à vérifier
            
        Returns:
            True si le MAC est valide
        """
        computed_mac = self.compute_mac(message)
        return computed_mac == mac


if __name__ == "__main__":
    # Test de chiffrement
    print("=== Test de chiffrement ===")
    crypto = RadioCrypto(password="airsoft2024")
    
    # Chiffre un message texte
    message = b"Message radio secret pour equipe Alpha"
    print(f"Message original: {message.decode()}")
    
    iv, ciphertext, mac = crypto.encrypt(message)
    print(f"IV: {iv.hex()[:32]}...")
    print(f"Taille chiffrée: {len(ciphertext)} bytes")
    
    # Déchiffre
    decrypted = crypto.decrypt(iv, ciphertext, mac)
    print(f"Message déchiffré: {decrypted.decode()}")
    print(f"Succès: {message == decrypted}")
    
    # Test avec array NumPy
    print("\n=== Test avec données NumPy ===")
    data = np.random.randint(0, 2, size=(10, 10))
    print(f"Shape originale: {data.shape}")
    
    iv, ciphertext, mac, shape, dtype = crypto.encrypt_array(data)
    print(f"Taille chiffrée: {len(ciphertext)} bytes")
    
    recovered = crypto.decrypt_array(iv, ciphertext, mac, shape, dtype)
    print(f"Données récupérées correctement: {np.array_equal(data, recovered)}")
    
    # Test d'authentification
    print("\n=== Test d'authentification ===")
    auth = MessageAuthenticator()
    
    msg = b"Message authentifie"
    mac = auth.compute_mac(msg)
    print(f"MAC: {mac.hex()[:32]}...")
    print(f"Vérification valide: {auth.verify_mac(msg, mac)}")
    print(f"Vérification invalide: {auth.verify_mac(b'Message modifie', mac)}")
