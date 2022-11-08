import numpy as np
import matplotlib.pyplot as plt
import struct

epochs = 100000
single_file = open("single_loss","rb")
single_loss = struct.unpack("d"*epochs, single_file.read(epochs*8))
single_x = np.arange(1, len(single_loss)+1)


record_epochs = [0]
records = [np.inf]
for epoch in range(epochs):
    if single_loss[epoch] < records[-1]:
        records.append(single_loss[epoch])
        record_epochs.append(epoch+1)

plt.loglog(single_x, single_loss, alpha=0.3, label = "single losses")
plt.loglog(record_epochs, records, '.--', alpha=1.0, label = "single records")

plt.grid(alpha = 0.5)
plt.legend()
plt.xlabel("Epoch")
plt.ylabel("Loss")

plt.savefig("single.png")
plt.show()
