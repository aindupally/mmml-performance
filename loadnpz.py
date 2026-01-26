import numpy as np

# Load the NPZ file
npz_file = np.load('data/validation_dataset.npz')

# Display all arrays in the file
print("Arrays in the NPZ file:")
for key in npz_file.files:
    print(f"  {key}: shape={npz_file[key].shape}, dtype={npz_file[key].dtype}")

# Access specific arrays
for key in npz_file.files:
    print(f"\n{key}:")
    print(npz_file[key])

# Close the file
npz_file.close()