import numpy as np

# Path to your .npy file
file_path = 'tpose/h1_tpose.npy'

# Load the .npy file
data = np.load(file_path, allow_pickle=True)


print(data)

