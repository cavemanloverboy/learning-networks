import struct
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

emulator_epochs = 10**7
emulator_file = open("emulator_losses","rb")
emulator_loss = struct.unpack("d"*emulator_epochs, emulator_file.read(emulator_epochs*8))
emulator_x = np.arange(1, len(emulator_loss)+1)
emulator_smooth = np.exp(savgol_filter(np.log(emulator_loss), 1001, 3))

monolithic_epochs = 10**7
monolithic_file = open("128_4_em","rb")
monolithic_loss = struct.unpack("d"*monolithic_epochs, monolithic_file.read(monolithic_epochs*8))
monolithic_x = np.arange(1, len(monolithic_loss)+1)
monolithic_smooth = np.exp(savgol_filter(np.log(monolithic_loss), 1001, 3))


plt.loglog(emulator_x, emulator_loss, "C0", alpha=0.2)
plt.loglog(monolithic_x, monolithic_loss, "C1", alpha=0.2)
plt.loglog(emulator_x, emulator_smooth, "C0", alpha=1.0,label = "learned")
plt.loglog(monolithic_x, monolithic_smooth, "C1", alpha=0.9,label = "monolithic")
plt.legend()

plt.grid(alpha = 0.5)
plt.xlabel("Epoch")
plt.ylabel("Loss")

plt.savefig("em_comparison.png")
