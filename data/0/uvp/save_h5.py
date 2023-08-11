import h5py
import numpy as np

u = np.load("u.npy")
v = np.load("v.npy")
p = np.load("p.npy")

# u = np.random.random((100, 150, 170))
# v = np.random.random((100, 150, 170))
# p = np.random.random((100, 150, 170))

combined_data = np.stack([u, v, p], axis=0)

# Sample metadata
xlims = [-0.35, 2]
ylims = [-0.35, 0.35]

with h5py.File('uvp.hdf5', 'w') as f:
    # Create a group for the datasets
    group = f.create_group("data_group")
    
    # Attach metadata as attributes to the group
    group.attrs['xlims'] = xlims
    group.attrs['ylims'] = ylims

    # Create datasets within the group
    dset_combined = group.create_dataset('uvp', data=combined_data, compression="lzf")