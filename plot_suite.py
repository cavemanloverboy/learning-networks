import numpy as np
import matplotlib.pyplot as plt
from glob import glob
import struct

files = np.sort(glob("single_*_*"))
EPOCHS = 100000

plt.figure()
for filename in files:
    (width, depth) = filename.split("_")[1:]
    if int(depth) != 2:
        continue
    file = open(filename,"rb")
    loss =  struct.unpack("d"*EPOCHS, file.read(EPOCHS*8))
    file.close()
    epochs = np.arange(1, len(loss)+1)

    plt.loglog(epochs, loss, alpha=0.3, label = filename[7:])

plt.legend()
plt.grid(alpha = 0.5)
plt.xlabel("Epoch")
plt.ylabel("Loss")

plt.savefig("suite.png")
plt.show()
