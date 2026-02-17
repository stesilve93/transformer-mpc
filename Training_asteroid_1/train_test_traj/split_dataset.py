import numpy as np

# === Data loading ===
data = np.load('dataset_nominal_NO_ROT.npz')
trajectories = np.transpose(data['trajectories'], (1, 2, 0))  # (N_instants, 6, N_tot)
controls     = np.transpose(data['controls'],     (1, 2, 0))  # (N_instants, 3, N_tot)

# === Data shuffle ===
n_samples = trajectories.shape[2]
perm = np.random.permutation(n_samples)
trajectories = trajectories[:, :, perm]
controls     = controls[:, :, perm]

# === Splitting train/test ===
# tot traj inspection --> 120_000
n_train = 100_000
X_train, X_test = trajectories[:, :, :n_train], trajectories[:, :, n_train:]
U_train, U_test = controls[:, :, :n_train], controls[:, :, n_train:]

# === Salvataggio ===
np.save('trajectories_train.npy', X_train)
np.save('trajectories_test.npy', X_test)
np.save('control_train.npy', U_train)
np.save('control_test.npy', U_test)

print("Data saved!")
