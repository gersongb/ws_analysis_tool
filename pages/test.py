import h5py
import os

h5_path = r"C:\Users\g.garsed\Documents\wt_tool_testing\Cadillac_10-2025_windshear_GGB\data\Cadillac_10-2025_windshear_GGB.h5"

if not os.path.exists(h5_path):
    print(f"HDF5 file not found: {h5_path}")
else:
    with h5py.File(h5_path, "a") as h5f:
        if "wt_runs" in h5f:
            wt_runs = h5f["wt_runs"]
            if "Run00005" in wt_runs:
                del wt_runs["Run00005"]
                print("Deleted group 'Run00005' from wt_runs.")
            else:
                print("Group 'Run00005' does not exist in wt_runs.")
        else:
            print("Group 'wt_runs' does not exist in the HDF5 file.")