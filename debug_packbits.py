import numpy as np
from src.crypto import RadioCrypto

np.random.seed(0)
plain_bits = np.random.randint(0,2, size=256, dtype=np.uint8)
packed = np.packbits(plain_bits, bitorder='little')

iv, ct, mac = RadioCrypto(password='airsoft2024').encrypt(packed.tobytes())

ct_bits = np.unpackbits(np.frombuffer(ct, dtype=np.uint8), bitorder='little')
ct_repacked = np.packbits(ct_bits.astype(np.uint8), bitorder='little').tobytes()

print('bytes_equal=', ct == ct_repacked)
