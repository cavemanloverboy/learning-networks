import numpy as np
import matplotlib.pyplot as plt
import struct

epochs = 100000
double_file = open("double_loss","rb")
double_loss = struct.unpack("d"*epochs, double_file.read(epochs*8))
double_x = np.arange(1, len(double_loss)+1)
double_record_epochs = [0]
double_records = [np.inf]
for epoch in range(epochs):
    if double_loss[epoch] < double_records[-1]:
        double_records.append(double_loss[epoch])
        double_record_epochs.append(epoch+1)

single_file = open("single_loss","rb")
single_loss = struct.unpack("d"*epochs, single_file.read(epochs*8))
single_x = np.arange(1, len(single_loss)+1)
single_record_epochs = [0]
single_records = [np.inf]
for epoch in range(epochs):
    if single_loss[epoch] < single_records[-1]:
        single_records.append(single_loss[epoch])
        single_record_epochs.append(epoch+1)

plt.loglog(double_x, double_loss, alpha=0.3, label = "double")
plt.loglog(single_x, single_loss, alpha=0.3, label = "single")
plt.loglog(double_record_epochs, double_records, 'C0.--', alpha=1.0, label = "double records")
plt.loglog(single_record_epochs, single_records, 'C1.--', alpha=1.0, label = "single records")


plt.grid(alpha = 0.5)
plt.legend()
plt.xlabel("Epoch")
plt.ylabel("Loss")

plt.savefig("single_vs_double.png")
plt.show()
