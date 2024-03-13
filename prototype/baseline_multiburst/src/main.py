import os
import sys

filename = "results/foo.txt"
os.makedirs(os.path.dirname(filename), exist_ok=True)
with open(filename, "w") as f:
    f.write("FOOBAR")
f.close()
sys.exit()