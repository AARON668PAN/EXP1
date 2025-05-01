import numpy as np

# # Path to your .npy file
# file_path = 'data/retarget_npy/NN_retarget/retarget_final0.npy'

# # Load the .npy file
# data = np.load(file_path, allow_pickle=True).item()
# data = data['root_translation']['arr']

# print(data.shape)



# Path to your .npy file
# file_path = 'data/raw_data/all_train_data/raw_data_all.npy'

# # Load the .npy file
# data = np.load(file_path, allow_pickle=True).item()


# data = data['motion_cont6d']

# print(data.shape)



file_path = 'data/processed_data/g1_dof29_data.npy'

# Load the .npy file
data = np.load(file_path, allow_pickle=True)



print(data)
