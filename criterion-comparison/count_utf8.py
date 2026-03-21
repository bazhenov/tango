import sys
from collections import Counter

with open(sys.argv[1], "rb") as f:
    data = f.read()

counts = Counter()
i = 0
while i < len(data):
    b = data[i]
    if b < 0x80:
        size = 1
    elif b < 0xE0:
        size = 2
    elif b < 0xF0:
        size = 3
    else:
        size = 4
    counts[size] += 1
    i += size

for size in (1, 2, 3, 4):
    print(f"{size}-byte: {counts[size]}")
