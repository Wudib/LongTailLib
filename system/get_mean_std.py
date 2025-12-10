from statistics import mean
import numpy as np
import os
import sys

# Get filename from user input
file_name = input("Enter filename (without .out extension): ") + '.out'

# Check if file exists
if not os.path.exists(file_name):
    print(f"Error: File '{file_name}' not found!")
    sys.exit(1)

acc = []

try:
    with open(file_name, 'r') as f:
        is_best = False
        for l in f.readlines():
            if is_best:
                try:
                    acc.append(float(l.strip()))
                except ValueError:
                    print(f"Warning: Could not parse line as float: {l.strip()}")
                is_best = False
            elif ('Best local accuracy' in l) or ('Best accuracy' in l):
                is_best = True
except Exception as e:
    print(f"Error reading file: {e}")
    sys.exit(1)

# Check if we found any accuracy values
if len(acc) == 0:
    print("Warning: No accuracy values found in the file!")
    print("Make sure the file contains lines with 'Best local accuracy' or 'Best accuracy'")
    sys.exit(1)

# Print results
print(f"\nFound {len(acc)} accuracy values:")
print(acc)
print(f"\nMean accuracy: {mean(acc)*100:.2f}%")
print(f"Std deviation: {np.std(acc)*100:.2f}%")
