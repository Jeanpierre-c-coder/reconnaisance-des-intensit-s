# -*- coding: utf-8 -*- 
"""
Repeat offline testing with Leave-One-Out strategy
Enhanced to scale the number of subjects dynamically
By Sébastien Mick, improved by Assistant
"""

# # IMPORTS
# - Built-in
from subprocess import call
# - Third-party
import numpy as np

# # CONSTANTS
# Original number of subjects
N_SUBJECTS = 18
# Multiplication factor for subjects
MULTIPLICATION_FACTOR = 2  # Multiply the number of subjects by this factor
# Total number of subjects after scaling
TOTAL_SUBJECTS = N_SUBJECTS * MULTIPLICATION_FACTOR

# Subprocess parameters
BASE_PARAMS = ["python3", "train_loo.py"]
# Whether to save mats
SAVE_MATS = True

# # METHODS

# # MAIN
if __name__ == "__main__":
    # Loop through all subjects
    for ind in range(TOTAL_SUBJECTS):
        params = BASE_PARAMS + [str(ind)]
        if SAVE_MATS:
            params.append("1")
        call(params)

    # Consolidate matrices if enabled
    if SAVE_MATS:
        loo_mats = []
        for ind in range(TOTAL_SUBJECTS):
            file_path = f"data/npy/loo_{ind}_mat.npy"
            try:
                mat = np.load(file_path)
                loo_mats.append(mat)
            except FileNotFoundError:
                print(f"Warning: File {file_path} not found. Skipping.")
        
        # Combine loaded matrices
        if loo_mats:
            mat_arr = np.array(loo_mats)
            print(f"Consolidated matrix shape: {mat_arr.shape}")
            np.save("data/npy/all_loo_mat.npy", mat_arr)
        else:
            print("No matrices were loaded. Exiting.")
