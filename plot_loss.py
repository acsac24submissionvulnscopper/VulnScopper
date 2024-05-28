import re
import matplotlib.pyplot as plt


PATTERN = r'average binary cross entropy: (\d+\.\d+)'  # depends on the need to extract the loss value


file_name = '/home/ULTRA/output/Ultra/JointDataset/2024-01-27-08-27-51/log.txt'

with open(file_name, 'r') as f:
    lines = f.readlines()

lines = [l for l in lines if re.search(PATTERN, l)]
lines = [float(re.search(PATTERN, l).group(1)) for l in lines]


# show all xticks
plt.plot(lines)
plt.xticks(range(len(lines)))
plt.savefig('loss.png')