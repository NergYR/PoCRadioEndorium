import numpy as np
from src.simulator import RadioSimulator, RadioConfig
from src.propagation import Environment

cfg = RadioConfig(encryption_enabled=True, use_dsss=True, environment=Environment.OUTDOOR_OPEN)
sim = RadioSimulator(cfg)

# Packet
data = np.random.randint(0,2, size=256, dtype=np.uint8)

signal, meta = sim.transmit_packet(data)
# Pas de bruit: on passe directement au récepteur
rx = sim.receive_packet(signal, meta)

print('ok=', np.array_equal(rx, data), 'len=', len(rx))
