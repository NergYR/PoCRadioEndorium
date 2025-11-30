import numpy as np
from src.modulation import DSSSModulator

np.random.seed(0)
# random bits simulating ciphertext bits
bits = np.random.randint(0,2,size=256, dtype=np.uint8)
mod = DSSSModulator(chip_rate=1000000, data_rate=10000)
spread = mod.spread(bits)
recovered = mod.despread(spread)
print('equal=', np.array_equal(bits, recovered), 'errors=', np.sum(bits!=recovered))
