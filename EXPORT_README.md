# D1 Processed Data Export Tool

Standalone script to export d1_processed data from HDF5 files to CSV format.

The d1_processed dataset includes:
- **Map setpoints**: setpoint, road_speed, wind_speed, frh, rrh, roll, yaw, w_cz, w_cx
- **D1 data columns**: D, L, LF, LR, DYNPR, Tunnel_Air_Temp, Relative_Humidity
- **Calculated columns**: CL, CD, CLW, CDW

## Quick Start

### Export ALL runs (recommended)
```bash
python export_d1_to_csv.py "path/to/homologation.h5"
```

### Export a specific run
```bash
python export_d1_to_csv.py "path/to/homologation.h5" "run_name"
```

### Export with custom output path
```bash
python export_d1_to_csv.py "path/to/homologation.h5" "run_name" "custom_output.csv"
```

### List available runs
```bash
python export_d1_to_csv.py --list "path/to/homologation.h5"
```

## Usage Examples

### Example 1: List all runs
```bash
python export_d1_to_csv.py --list "C:/Users/g.garsed/Documents/dev/homologation.h5"
```

Output:
```
Available runs in C:/Users/g.garsed/Documents/dev/homologation.h5:
  - 251020_071654_Run0002
  - 251020_080530_Run0003
  - 251020_091245_Run0004
```

### Example 2: Export ALL runs
```bash
python export_d1_to_csv.py "C:/Users/g.garsed/Documents/dev/homologation.h5"
```

Output:
```
No run name specified. Exporting all runs from HDF5 file...

Available runs in C:/Users/g.garsed/Documents/dev/homologation.h5:
  - 251020_071654_Run0002
  - 251020_080530_Run0003
  - 251020_091245_Run0004

============================================================
Exporting 3 run(s)...
============================================================

[1/3] Exporting 251020_071654_Run0002...
[OK] Successfully exported d1_processed data to: ...

[2/3] Exporting 251020_080530_Run0003...
[OK] Successfully exported d1_processed data to: ...

[3/3] Exporting 251020_091245_Run0004...
[OK] Successfully exported d1_processed data to: ...

============================================================
Export Summary:
  Successful: 3/3
  Failed: 0/3
============================================================
```

All files saved to: `C:/Users/g.garsed/Documents/dev/d1_exports/`

### Example 3: Export a specific run
```bash
python export_d1_to_csv.py "C:/Users/g.garsed/Documents/dev/homologation.h5" "251020_071654_Run0002"
```

Output file: `C:/Users/g.garsed/Documents/dev/d1_exports/251020_071654_Run0002_d1_processed.csv`

**Note:** CSV files are automatically saved in a `d1_exports` folder within the same directory as the HDF5 file.

### Example 4: Export with custom output name
```bash
python export_d1_to_csv.py "C:/Users/g.garsed/Documents/dev/homologation.h5" "251020_071654_Run0002" "my_export.csv"
```

Output file: `my_export.csv`

## Output Folder Structure

By default, CSV files are saved in a `d1_exports` folder within the homologation directory:

```
homologation_folder/
├── homologation.h5
└── d1_exports/              ← Auto-created
    ├── 251020_071654_Run0002_d1_processed.csv
    ├── 251020_080530_Run0003_d1_processed.csv
    └── 251020_091245_Run0004_d1_processed.csv
```

You can override this by specifying a custom output path as the third argument.

## CSV Output Format

The exported CSV file contains:
- **Header row**: Column names from the d1_processed dataset
- **Data rows**: All setpoint measurements with combined data

Example CSV structure:
```csv
setpoint,road_speed,wind_speed,frh,rrh,roll,yaw,w_cz,w_cx,D,L,LF,LR,DYNPR,Tunnel_Air_Temp,Relative_Humidity,CL,CD,CLW,CDW
First,60,60,50,80,0,0,,,123.45,678.90,45.6,56.7,234.5,25.3,45.2,,,
S01,60,60,50,80,1.2,0,,,,125.30,682.15,46.1,57.2,235.1,25.4,45.3,0.245,0.123,0.0,0.0
S02,60,60,50,80,1.2,-5,,,128.50,685.20,46.5,58.0,236.0,25.5,45.4,0.248,0.125,2.48,0.0
...
```

**Columns included:**
- **Map setpoints**: setpoint, road_speed, wind_speed, frh, rrh, roll, yaw, w_cz, w_cx
- **D1 measurements**: D, L, LF, LR, DYNPR, Tunnel_Air_Temp, Relative_Humidity
- **Calculated coefficients**: CL, CD, CLW, CDW

## Requirements

- Python 3.x
- h5py library

Install requirements:
```bash
pip install h5py
```

## Error Handling

The script will display helpful error messages if:
- HDF5 file not found
- Run name doesn't exist
- d1 dataset not found
- Invalid file format

## Integration with Batch Scripts

### Option 1: Export all runs (Recommended)

```batch
@echo off
set H5_FILE="C:/path/to/homologation.h5"

REM Export all runs automatically
python export_d1_to_csv.py %H5_FILE%

echo All exports completed!
pause
```

### Option 2: Export specific runs

```batch
@echo off
set H5_FILE="C:/path/to/homologation.h5"

REM Export specific runs to default location (d1_exports folder)
python export_d1_to_csv.py %H5_FILE% "251020_071654_Run0002"
python export_d1_to_csv.py %H5_FILE% "251020_080530_Run0003"
python export_d1_to_csv.py %H5_FILE% "251020_091245_Run0004"

echo All exports completed!
echo Files saved to: C:/path/to/d1_exports/
```

## Programmatic Usage

You can also import and use the script in your own Python code:

```python
from export_d1_to_csv import export_d1_to_csv, export_all_runs

# Option 1: Export all runs
success = export_all_runs(h5_path="C:/path/to/homologation.h5")

# Option 2: Export a specific run
success = export_d1_to_csv(
    h5_path="C:/path/to/homologation.h5",
    run_name="251020_071654_Run0002",
    output_csv="output.csv"  # Optional
)

if success:
    print("Export successful!")
else:
    print("Export failed")
```
