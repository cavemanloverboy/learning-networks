import numpy as np
import matplotlib.pyplot as plt
from glob import glob
import struct

files = np.sort(glob("gen_train_losses_*_*"))
EPOCHS = 100000
EPOCHS = 1000

plt.figure()
for filename in files:
    (width, depth) = filename.split("_")[3:]
    file = open(filename,"rb")
    loss =  struct.unpack("d"*EPOCHS, file.read(EPOCHS*8))
    file.close()
    epochs = np.arange(1, len(loss)+1)

    plt.loglog(epochs, [l for l in loss], alpha=0.3, label = filename[7:])

plt.legend()
plt.grid(alpha = 0.5)
plt.xlabel("Epoch")
plt.ylabel("Loss")

plt.savefig("suite.png")
plt.show()
