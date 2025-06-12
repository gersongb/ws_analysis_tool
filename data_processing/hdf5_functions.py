import h5py
import json

def create_homologation_hdf5(h5_path, setup_data):
    """
    Create an HDF5 file for a homologation session.
    - h5_path: path to the HDF5 file to create
    - setup_data: dict of setup/meta data to store as attributes
    """
    with h5py.File(h5_path, "w") as h5f:
        for k, v in setup_data.items():
            # Store only simple types as attributes
            if isinstance(v, (str, int, float)):
                h5f.attrs[k] = v
        # Store the full setup JSON as a string attribute
        h5f.attrs['setup_json'] = json.dumps(setup_data)
        h5f.create_group("wt_runs")
