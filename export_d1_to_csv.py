#!/usr/bin/env python3
"""
Standalone script to export d1_processed data from HDF5 file to CSV.

The d1_processed dataset includes:
- Map setpoints (setpoint, road_speed, wind_speed, frh, rrh, roll, yaw, w_cz, w_cx)
- D1 data columns (D, L, LF, LR, DYNPR, Tunnel_Air_Temp, Relative_Humidity)
- Calculated columns (CL, CD, CLW, CDW)

CSV files are saved in a 'd1_exports' folder within the homologation directory by default.

Usage:
    python export_d1_to_csv.py <h5_file_path> [run_name] [output_csv_path]

Examples:
    # Export ALL runs to default location
    python export_d1_to_csv.py "C:/path/to/homologation.h5"
    
    # Export a specific run to default location
    python export_d1_to_csv.py "C:/path/to/homologation.h5" "251020_071654_Run0002"
    
    # Export a specific run to custom location
    python export_d1_to_csv.py "C:/path/to/homologation.h5" "251020_071654_Run0002" "output.csv"
    
    # List available runs
    python export_d1_to_csv.py --list "C:/path/to/homologation.h5"
"""

import sys
import os
import h5py
import csv


def export_d1_to_csv(h5_path, run_name, output_csv=None):
    """
    Export d1_processed dataset from HDF5 file to CSV.
    
    Args:
        h5_path: Path to the HDF5 file
        run_name: Name of the run (e.g., "251020_071654_Run0002")
        output_csv: Optional output CSV path. If None, saves to <homologation_dir>/d1_exports/<run_name>_d1_processed.csv
    
    Returns:
        True if successful, False otherwise
    """
    # Validate inputs
    if not os.path.exists(h5_path):
        print(f"Error: HDF5 file not found: {h5_path}")
        return False
    
    # Default output path: create "d1_exports" folder in the same directory as the HDF5 file
    if output_csv is None:
        h5_dir = os.path.dirname(os.path.abspath(h5_path))
        exports_dir = os.path.join(h5_dir, "d1_exports")
        
        # Create exports directory if it doesn't exist
        if not os.path.exists(exports_dir):
            os.makedirs(exports_dir)
            print(f"Created exports directory: {exports_dir}")
        
        output_csv = os.path.join(exports_dir, f"{run_name}_d1_processed.csv")
    
    try:
        # Open HDF5 file
        with h5py.File(h5_path, 'r') as h5f:
            # Check if wt_runs group exists
            if "wt_runs" not in h5f:
                print(f"Error: 'wt_runs' group not found in {h5_path}")
                return False
            
            wt_runs = h5f["wt_runs"]
            
            # Check if run exists
            if run_name not in wt_runs:
                print(f"Error: Run '{run_name}' not found in HDF5 file")
                print(f"Available runs: {list(wt_runs.keys())}")
                return False
            
            run_grp = wt_runs[run_name]
            
            # Check if d1_processed dataset exists
            if "d1_processed" not in run_grp:
                print(f"Error: 'd1_processed' dataset not found for run '{run_name}'")
                print(f"Available datasets: {list(run_grp.keys())}")
                return False
            
            d1_processed_dataset = run_grp["d1_processed"]
            
            # Get data
            d1_data = d1_processed_dataset[:]
            
            # Get column names
            columns = []
            if "columns" in d1_processed_dataset.attrs:
                columns = [
                    col.decode() if isinstance(col, bytes) else col 
                    for col in d1_processed_dataset.attrs["columns"]
                ]
            else:
                # Generate default column names if not found
                columns = [f"col_{i}" for i in range(d1_data.shape[1])]
            
            # Write to CSV
            with open(output_csv, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                
                # Write header
                writer.writerow(columns)
                
                # Write data rows - decode byte strings to regular strings
                for row in d1_data:
                    # Decode each cell if it's a byte string
                    decoded_row = []
                    for cell in row:
                        if isinstance(cell, bytes):
                            decoded_row.append(cell.decode('utf-8'))
                        else:
                            decoded_row.append(cell)
                    writer.writerow(decoded_row)
            
            print(f"[OK] Successfully exported d1_processed data to: {output_csv}")
            print(f"  Run: {run_name}")
            print(f"  Rows: {d1_data.shape[0]}")
            print(f"  Columns: {len(columns)}")
            print(f"  Dataset: d1_processed (includes map setpoints + d1 data + calculated columns)")
            
            return True
            
    except Exception as e:
        print(f"Error exporting d1 data: {e}")
        return False


def list_runs(h5_path):
    """List all available runs in the HDF5 file."""
    if not os.path.exists(h5_path):
        print(f"Error: HDF5 file not found: {h5_path}")
        return []
    
    try:
        with h5py.File(h5_path, 'r') as h5f:
            if "wt_runs" not in h5f:
                print(f"No 'wt_runs' group found in {h5_path}")
                return []
            
            runs = list(h5f["wt_runs"].keys())
            if runs:
                print(f"\nAvailable runs in {h5_path}:")
                for run in runs:
                    print(f"  - {run}")
            else:
                print(f"No runs found in {h5_path}")
            return runs
    
    except Exception as e:
        print(f"Error reading HDF5 file: {e}")
        return []


def export_all_runs(h5_path):
    """Export all runs from the HDF5 file to CSV."""
    runs = list_runs(h5_path)
    
    if not runs:
        print("No runs found to export.")
        return False
    
    print(f"\n{'='*60}")
    print(f"Exporting {len(runs)} run(s)...")
    print(f"{'='*60}\n")
    
    success_count = 0
    failed_count = 0
    
    for i, run_name in enumerate(runs, 1):
        print(f"[{i}/{len(runs)}] Exporting {run_name}...")
        success = export_d1_to_csv(h5_path, run_name)
        
        if success:
            success_count += 1
        else:
            failed_count += 1
        print()  # Blank line between exports
    
    # Summary
    print(f"{'='*60}")
    print(f"Export Summary:")
    print(f"  Successful: {success_count}/{len(runs)}")
    print(f"  Failed: {failed_count}/{len(runs)}")
    print(f"{'='*60}")
    
    return failed_count == 0


def main():
    """Main entry point for command-line usage."""
    if len(sys.argv) < 2:
        print(__doc__)
        print("\nOptions:")
        print("  --list <h5_file_path>    List all available runs in the HDF5 file")
        sys.exit(1)
    
    # Check for --list option
    if sys.argv[1] == "--list":
        if len(sys.argv) < 3:
            print("Error: --list requires HDF5 file path")
            sys.exit(1)
        list_runs(sys.argv[2])
        sys.exit(0)
    
    # Export mode
    h5_path = sys.argv[1]
    
    if len(sys.argv) < 3:
        # No run name specified - export all runs
        print("No run name specified. Exporting all runs from HDF5 file...")
        success = export_all_runs(h5_path)
        sys.exit(0 if success else 1)
    
    run_name = sys.argv[2]
    output_csv = sys.argv[3] if len(sys.argv) > 3 else None
    
    # Export single run
    success = export_d1_to_csv(h5_path, run_name, output_csv)
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
