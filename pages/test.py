import h5py
import os

h5_path = r"C:\Users\g.garsed\Documents\wt_tool_testing\Cadillac_10-2025_windshear_GGB\data\Cadillac_10-2025_windshear_GGB.h5"

if not os.path.exists(h5_path):
    # HDF5 file not found
    pass
else:
    with h5py.File(h5_path, "a") as h5f:
        if "wt_runs" in h5f:
            wt_runs = h5f["wt_runs"]
            if "Run00005" in wt_runs:
                del wt_runs["Run00005"]
                # Deleted group 'Run00005' from wt_runs.
            else:
                # Group 'Run00005' does not exist in wt_runs.
                pass
                
        else:
            # Group 'wt_runs' does not exist in the HDF5 file.
            pass