import os

import dash
from dash import html, dcc, Output, Input, State, callback_context
from dash.dependencies import ALL, Input, Output, State
import h5py
import numpy as np
import json
from dash import no_update
from dash import dash_table

# ---- Unit Conversion Constants (Wind Tunnel) ----
# Pressure: pounds per square foot (psf) to Pascal (Pa)
PSF_TO_PA = 47.88125
# Force: pound-force (lbf) to Newton (N)
LBF_TO_NEWTONS = 4.44822

def _get_attr_str(arr):
    return [v.decode() if isinstance(v, (bytes, bytearray)) else v for v in arr]

def compute_weighted_coeffs_from_d1_processed(run_grp):
    """Compute weighted_Cz and weighted_Cx based on d1_processed dataset and store as run attributes.

    Given:
    - drag_tare = D value at setpoint == 'ZeroD1'
    - lift_tare = L value at setpoint == 'ZeroL'
    - CL = (L - lift_tare) * LBF_TO_NEWTONS / (DYNPR * PSF_TO_PA)
    - CLW = CL * w_cz (only for valid measurements)
      weighted_Cz = sum(CLW) / sum(ALL w_cz from map)
    - CD = (D - drag_tare) * LBF_TO_NEWTONS / (DYNPR * PSF_TO_PA)
    - min_CD = minimum CD from rows with valid (non-NaN) w_cx
    - CDW = CD * w_cx (only for valid measurements)
      weighted_Cx = sum(CDW) / sum(ALL w_cx from map) + 0.2 * min_CD
    
    Note: Weight sums include ALL valid weights from the map, not just those with valid measurements.
    """
    try:
        if "d1_processed" not in run_grp:
            return False, "d1_processed not found"
        ds = run_grp["d1_processed"]
        data = ds[:]
        cols = []
        if "columns" in ds.attrs:
            cols = _get_attr_str(ds.attrs["columns"])
        if not cols or data.size == 0:
            return False, "d1_processed has no data or columns"

        # Resolve indices (case-insensitive)
        sp_idx = _find_col_index('setpoint', cols)
        L_idx = _find_col_index('L', cols)
        D_idx = _find_col_index('D', cols)
        DYNPR_idx = _find_col_index('DYNPR', cols)
        wcz_idx = _find_col_index('w_cz', cols)
        wcx_idx = _find_col_index('w_cx', cols)

        for required_name, idx in [("setpoint", sp_idx), ("L", L_idx), ("DYNPR", DYNPR_idx), ("w_cz", wcz_idx)]:
            if idx == -1:
                return False, f"Missing required column '{required_name}' in d1_processed"

        # Helper to read cell as float when possible
        def to_float(x):
            try:
                # Handle bytes and numpy types
                if isinstance(x, (bytes, bytearray)):
                    x = x.decode()
                return float(x)
            except Exception:
                return float('nan')

        # Find tares
        drag_tare = float('nan')
        lift_tare = float('nan')
        for r in range(data.shape[0]):
            sp_val = data[r, sp_idx]
            if isinstance(sp_val, (bytes, bytearray)):
                sp_val = sp_val.decode()
            if str(sp_val) == 'ZeroD1' and D_idx != -1:
                drag_tare = to_float(data[r, D_idx])
            if str(sp_val) == 'ZeroL':
                lift_tare = to_float(data[r, L_idx])
        # First pass: find minimum CD from rows with valid w_cx
        min_cd = float('inf')
        min_cd_setpoint = "Unknown"
        valid_cd_values = []
        for r in range(data.shape[0]):
            D = to_float(data[r, D_idx]) if D_idx != -1 else float('nan')
            dynpr = to_float(data[r, DYNPR_idx])
            wcx = to_float(data[r, wcx_idx]) if wcx_idx != -1 else float('nan')
            
            if not (np.isnan(D) or np.isnan(drag_tare) or np.isnan(dynpr) or np.isnan(wcx) or dynpr == 0):
                cd = (D - drag_tare) * LBF_TO_NEWTONS / (dynpr * PSF_TO_PA)
                if not np.isnan(cd):
                    valid_cd_values.append(cd)
                    if cd < min_cd:
                        min_cd = cd
                        # Get setpoint name for this row
                        sp_val = data[r, sp_idx]
                        if isinstance(sp_val, (bytes, bytearray)):
                            sp_val = sp_val.decode()
                        min_cd_setpoint = str(sp_val)
        
        # Handle case where no valid CD values found
        if min_cd == float('inf'):
            min_cd = 0.0
            min_cd_setpoint = "None"
        
        print(f"[WT] Minimum CD found (from rows with valid w_cx): {min_cd:.6f} at setpoint '{min_cd_setpoint}'")
        
        # Second pass: compute CL, CD, CLW and CDW, accumulate sums and sum of weights
        clw_sum = 0.0
        cdw_sum = 0.0
        wcz_sum = 0.0
        wcx_sum = 0.0
        for r in range(data.shape[0]):
            L = to_float(data[r, L_idx])
            D = to_float(data[r, D_idx]) if D_idx != -1 else float('nan')
            dynpr = to_float(data[r, DYNPR_idx])
            wcz = to_float(data[r, wcz_idx])
            wcx = to_float(data[r, wcx_idx]) if wcx_idx != -1 else float('nan')
            
            # Accumulate weight sums for ALL valid weights (not just when measurements are valid)
            if not np.isnan(wcz):
                wcz_sum += wcz
            if not np.isnan(wcx):
                wcx_sum += wcx
            
            # Calculate CL (Coefficient of Lift) first
            if not (np.isnan(L) or np.isnan(lift_tare) or np.isnan(dynpr) or dynpr == 0):
                cl = (L - lift_tare) * LBF_TO_NEWTONS / (dynpr * PSF_TO_PA)
                if not np.isnan(cl) and not np.isnan(wcz):
                    clw = cl * wcz  # CLW = CL * w_cz
                    clw_sum += clw
            
            # Calculate CD (Coefficient of Drag) first
            if not (np.isnan(D) or np.isnan(drag_tare) or np.isnan(dynpr) or dynpr == 0):
                cd = (D - drag_tare) * LBF_TO_NEWTONS / (dynpr * PSF_TO_PA)
                if not np.isnan(cd) and not np.isnan(wcx):
                    cdw = cd * wcx  # CDW = CD * w_cx (back to original formula)
                    cdw_sum += cdw
        # Debug: print intermediate sums
        print(f"[WT DEBUG] clw_sum={clw_sum:.6f}, wcz_sum={wcz_sum:.6f}, cdw_sum={cdw_sum:.6f}, wcx_sum={wcx_sum:.6f}")
        
        # Final weighted values are normalized by the sum of weights
        weighted_cz = float(clw_sum / wcz_sum) if (wcz_sum != 0.0 and not np.isnan(wcz_sum)) else 0.0
        weighted_cx_base = float(cdw_sum / 100.0)  # Normalize by 100 instead of wcx_sum
        weighted_cx = weighted_cx_base + 0.2 * min_cd  # Add 20% of minimum CD to final weighted_Cx
        
        # Round to 3 decimal places (4th decimal will be zero)
        weighted_cz = round(weighted_cz, 3)
        weighted_cx = round(weighted_cx, 3)

        run_grp.attrs["weighted_Cz"] = weighted_cz
        run_grp.attrs["weighted_Cx"] = weighted_cx
        run_grp.attrs["min_CD"] = min_cd
        run_grp.attrs["min_CD_setpoint"] = min_cd_setpoint
        try:
            run_name = run_grp.name.split('/')[-1]
        except Exception:
            run_name = str(run_grp)
        print(f"[WT] Run '{run_name}': weighted_Cz={weighted_cz:.6f}, weighted_Cx_base={weighted_cx_base:.6f}, min_CD_offset={0.2 * min_cd:.6f}, weighted_Cx={weighted_cx:.6f}")
        
        # Return success message with minimum CD info
        success_msg = f"Minimum CD: {min_cd:.6f} at ride height '{min_cd_setpoint}'"
        return True, success_msg
    except Exception as e:
        return False, f"Error computing weighted coeffs: {e}"

def load_setpoints_from_map(map_name):
    """Load full setpoint data from the specified map in maps.json"""
    try:
        maps_config_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "config", "maps.json")
        )
        if not os.path.exists(maps_config_path):
            return [], []
        
        with open(maps_config_path, "r") as f:
            maps_data = json.load(f)
        
        if map_name in maps_data:
            setpoints_data = maps_data[map_name]
            if not setpoints_data:
                return [], []
            
            # Get column names from the first setpoint
            columns = list(setpoints_data[0].keys())
            
            # Create structured array data
            structured_data = []
            for setpoint in setpoints_data:
                row = []
                for col in columns:
                    value = setpoint.get(col)
                    # Handle None values by converting to empty string
                    if value is None:
                        row.append("")
                    else:
                        row.append(str(value))
                structured_data.append(row)
            
            return structured_data, columns
        else:
            return [], []
    except Exception as e:
        print(f"Error loading setpoints from map {map_name}: {e}")
        return [], []

def _normalize_colname(name: str) -> str:
    return ''.join(ch.lower() for ch in name.strip())

def _find_col_index(target: str, columns: list[str]) -> int:
    t = _normalize_colname(target)
    for i, c in enumerate(columns):
        if _normalize_colname(c) == t:
            return i
    return -1

def load_d1_structured(path):
    """Load d1 as a structured array preserving string columns (tab-delimited)."""
    try:
        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
        # Header is on line 3, data starts after line 4 (0-indexed)
        colnames = lines[3].strip().split('\t')
        import io
        data_lines = [line for i, line in enumerate(lines) if i > 4]
        # Use structured load with provided names and tab delimiter
        arr = np.genfromtxt(
            io.StringIO(''.join(data_lines)),
            delimiter='\t',
            names=colnames,
            dtype=None,
            encoding='utf-8',
            autostrip=True
        )
        return arr, colnames
    except Exception as e:
        print(f"Error load_d1_structured: {e}")
        return None, []

def create_d1_processed_data(map_name, d1_data, d1_columns, d1_structured=None, d1_structured_columns=None):
    """Create d1_processed data by combining map setpoints with d1 data columns"""
    try:
        # Load setpoints from map
        setpoints_data, map_columns = load_setpoints_from_map(map_name)
        if not setpoints_data or not map_columns:
            return [], []
        
        # Define the d1 columns we want to append
        d1_columns_to_append = ["D", "L", "LF", "LR", "DYNPR", "Tunnel_Air_Temp", "Relative_Humidity"]
        
        # Add calculated columns for CL, CD, CLW and CDW
        calculated_columns = ["CL", "CD", "CLW", "CDW"]
        
        # Debug: Print column information
        print(f"Available d1 columns: {d1_columns}")
        print(f"d1_data shape: {np.array(d1_data).shape if len(d1_data) > 0 else 'empty'}")
        
        # Find indices of the d1 columns we want
        d1_column_indices = {}
        for col in d1_columns_to_append:
            # Case-insensitive column name matching
            idx = _find_col_index(col, d1_columns)
            if idx != -1:
                d1_column_indices[col] = idx
                print(f"Found column {col} at index {idx}")
            else:
                print(f"Column {col} not found in d1_columns")
        
        # Combine map columns with d1 columns and calculated columns
        combined_columns = map_columns + d1_columns_to_append + calculated_columns
        
        # Create combined data
        combined_data = []

        # If we have a structured d1 with a setpoint-like column, compute per-setpoint values
        per_setpoint_values = {}
        if d1_structured is not None and d1_structured_columns:
            # Try to detect a setpoint column name
            setpoint_col_candidates = ["setpoint", "sp", "setpoint_name", "step", "set"]
            sp_idx = -1
            for cand in setpoint_col_candidates:
                sp_idx = _find_col_index(cand, d1_structured_columns)
                if sp_idx != -1:
                    break
            if sp_idx != -1:
                # Build per-setpoint aggregation
                # Access structured fields reliably by name
                sp_field = d1_structured_columns[sp_idx]
                # Prepare numeric column field names for extraction
                field_names = list(d1_structured.dtype.names) if hasattr(d1_structured, 'dtype') and d1_structured.dtype.names else None
                for sp_entry in setpoints_data:
                    sp_name = sp_entry[ _find_col_index('setpoint', map_columns) ] if _find_col_index('setpoint', map_columns) != -1 else None
                    vals = {}
                    if sp_name is not None and field_names and sp_field in field_names:
                        try:
                            mask = np.array([(str(v) == str(sp_name)) for v in d1_structured[sp_field]])
                            for col, idx in d1_column_indices.items():
                                # Find corresponding field in structured using case-insensitive match
                                fld_idx = _find_col_index(col, d1_structured_columns)
                                if fld_idx != -1:
                                    fld_name = d1_structured_columns[fld_idx]
                                    try:
                                        col_vals = []
                                        for v in d1_structured[fld_name][mask]:
                                            try:
                                                fv = float(v)
                                                if not np.isnan(fv):
                                                    col_vals.append(fv)
                                            except (ValueError, TypeError):
                                                pass
                                        vals[col] = float(np.mean(col_vals)) if col_vals else 0.0
                                    except Exception:
                                        vals[col] = 0.0
                        except Exception:
                            pass
                    per_setpoint_values[sp_name] = vals

        # Find tare values for CLW/CDW calculations
        drag_tare = 0.0
        lift_tare = 0.0
        
        # Helper function to safely convert to float
        def to_float(x):
            try:
                if isinstance(x, (bytes, bytearray)):
                    x = x.decode()
                return float(x)
            except Exception:
                return float('nan')
        
        # Look for tare values in per_setpoint_values or d1_data
        if per_setpoint_values:
            if 'ZeroD1' in per_setpoint_values and 'D' in per_setpoint_values['ZeroD1']:
                drag_tare = per_setpoint_values['ZeroD1']['D']
            if 'ZeroL' in per_setpoint_values and 'L' in per_setpoint_values['ZeroL']:
                lift_tare = per_setpoint_values['ZeroL']['L']
        else:
            # Fallback: look through setpoints_data for ZeroD1/ZeroL
            sp_col_idx = _find_col_index('setpoint', map_columns)
            if sp_col_idx != -1:
                for i, sp_row in enumerate(setpoints_data):
                    sp_name = sp_row[sp_col_idx]
                    if str(sp_name) == 'ZeroD1' and 'D' in d1_column_indices and i < len(d1_data):
                        drag_tare = to_float(d1_data[i][d1_column_indices['D']])
                    if str(sp_name) == 'ZeroL' and 'L' in d1_column_indices and i < len(d1_data):
                        lift_tare = to_float(d1_data[i][d1_column_indices['L']])

        # Find minimum CD from rows with valid w_cx for CDW calculation
        min_cd = float('inf')
        for i, setpoint_row in enumerate(setpoints_data):
            try:
                # Get D, DYNPR, and w_cx values for this setpoint
                D = 0.0
                dynpr = 0.0
                wcx = 0.0
                
                # Get setpoint name
                sp_name = None
                sp_col_idx = _find_col_index('setpoint', map_columns)
                if sp_col_idx != -1:
                    sp_name = setpoint_row[sp_col_idx]
                
                # Extract D and DYNPR values
                if sp_name is not None and sp_name in per_setpoint_values:
                    D = per_setpoint_values[sp_name].get('D', 0.0)
                    dynpr = per_setpoint_values[sp_name].get('DYNPR', 0.0)
                elif 'D' in d1_column_indices and 'DYNPR' in d1_column_indices and i < len(d1_data):
                    row_index = min(i, len(d1_data) - 1)
                    D = to_float(d1_data[row_index][d1_column_indices['D']])
                    dynpr = to_float(d1_data[row_index][d1_column_indices['DYNPR']])
                
                # Extract w_cx from map setpoint
                wcx_idx = _find_col_index('w_cx', map_columns)
                if wcx_idx != -1:
                    wcx = to_float(setpoint_row[wcx_idx])
                
                # Calculate CD if all values are valid
                if not (np.isnan(D) or np.isnan(drag_tare) or np.isnan(dynpr) or np.isnan(wcx) or dynpr == 0):
                    cd = (D - drag_tare) * LBF_TO_NEWTONS / (dynpr * PSF_TO_PA)
                    if not np.isnan(cd):
                        min_cd = min(min_cd, cd)
            except Exception:
                continue
        
        # Handle case where no valid CD values found
        if min_cd == float('inf'):
            min_cd = 0.0
        
        print(f"[WT] Minimum CD found for d1_processed (from rows with valid w_cx): {min_cd:.6f}")

        for i, setpoint_row in enumerate(setpoints_data):
            combined_row = setpoint_row.copy()

            # Determine values source: per-setpoint if available, else per-row i from d1_data
            sp_name = None
            sp_col_idx = _find_col_index('setpoint', map_columns)
            if sp_col_idx != -1:
                sp_name = setpoint_row[sp_col_idx]

            for col in d1_columns_to_append:
                val_str = ""
                used = False
                if sp_name is not None and sp_name in per_setpoint_values and col in per_setpoint_values[sp_name]:
                    val_str = str(per_setpoint_values[sp_name][col])
                    used = True
                elif col in d1_column_indices and len(d1_data) > 0:
                    # Map by row index i (clamped) to avoid repeating first row
                    row_index = min(i, len(d1_data) - 1)
                    col_index = d1_column_indices[col]
                    if col_index < len(d1_data[row_index]):
                        val_str = str(d1_data[row_index][col_index])
                        used = True
                combined_row.append(val_str)

            # Calculate CL, CD, CLW and CDW for this row
            cl_val = ""
            cd_val = ""
            clw_val = ""
            cdw_val = ""
            
            try:
                # Get values needed for calculations
                L = 0.0
                D = 0.0
                dynpr = 0.0
                wcz = 0.0
                wcx = 0.0
                
                # Extract L, D, DYNPR from the row we just built
                if sp_name is not None and sp_name in per_setpoint_values:
                    L = per_setpoint_values[sp_name].get('L', 0.0)
                    D = per_setpoint_values[sp_name].get('D', 0.0)
                    dynpr = per_setpoint_values[sp_name].get('DYNPR', 0.0)
                elif 'L' in d1_column_indices and 'D' in d1_column_indices and 'DYNPR' in d1_column_indices and i < len(d1_data):
                    row_index = min(i, len(d1_data) - 1)
                    L = to_float(d1_data[row_index][d1_column_indices['L']])
                    D = to_float(d1_data[row_index][d1_column_indices['D']])
                    dynpr = to_float(d1_data[row_index][d1_column_indices['DYNPR']])
                
                # Extract w_cz and w_cx from map setpoint
                wcz_idx = _find_col_index('w_cz', map_columns)
                wcx_idx = _find_col_index('w_cx', map_columns)
                if wcz_idx != -1:
                    wcz = to_float(setpoint_row[wcz_idx])
                if wcx_idx != -1:
                    wcx = to_float(setpoint_row[wcx_idx])
                
                # Calculate CL, CD, CLW and CDW using intermediate steps
                # Calculate CL (Coefficient of Lift) first
                if not np.isnan(L) and not np.isnan(lift_tare) and not np.isnan(dynpr) and dynpr != 0:
                    cl = (L - lift_tare) * LBF_TO_NEWTONS / (dynpr * PSF_TO_PA)
                    if not np.isnan(cl):
                        cl_val = f"{cl:.6f}"
                        if not np.isnan(wcz):
                            clw = cl * wcz  # CLW = CL * w_cz
                            clw_val = f"{clw:.6f}"
                
                # Calculate CD (Coefficient of Drag) first
                if not np.isnan(D) and not np.isnan(drag_tare) and not np.isnan(dynpr) and dynpr != 0:
                    cd = (D - drag_tare) * LBF_TO_NEWTONS / (dynpr * PSF_TO_PA)
                    if not np.isnan(cd):
                        cd_val = f"{cd:.6f}"
                        if not np.isnan(wcx):
                            cdw = cd * wcx  # CDW = CD * w_cx (back to original formula)
                            cdw_val = f"{cdw:.6f}"
                        
            except Exception as e:
                print(f"Error calculating CL/CD/CLW/CDW for row {i}: {e}")
            
            # Append calculated values in order: CL, CD, CLW, CDW
            combined_row.append(cl_val)
            combined_row.append(cd_val)
            combined_row.append(clw_val)
            combined_row.append(cdw_val)

            combined_data.append(combined_row)
        
        return combined_data, combined_columns
        
    except Exception as e:
        print(f"Error creating d1_processed data: {e}")
        return [], []

# Callback to show run folders in data source folder with Import buttons
@dash.callback(
    Output("run-folder-list", "children"),
    Input("current-homologation-store", "data"),
    Input("run-folder-refresh-interval", "n_intervals"),
)
def update_run_folder_list(homologation, n_intervals):
    from dash import html
    if not homologation or not homologation.get("data_source_folder"):
        return html.Div("No data source folder set.", style={"color": "#888"})
    folder = homologation["data_source_folder"]
    if not os.path.isdir(folder):
        return html.Div(f"Folder not found: {folder}", style={"color": "#c00"})
    try:
        subfolders = [f for f in os.listdir(folder) if os.path.isdir(os.path.join(folder, f))]
        subfolders = sorted(
            subfolders,
            key=lambda f: os.path.getmtime(os.path.join(folder, f)),
            reverse=True
        )
        return [
            html.Div([
                html.Button(
    "⬇️",  # Unicode down arrow as import icon
    id={"type": "import-run-btn", "index": f},
    n_clicks=0,
    title="Import",
    **{"data-folder": f},
    style={"marginRight": "10px", "fontSize": "1.3rem", "background": "none", "border": "none", "cursor": "pointer"}
),
                html.Span(f)
            ], style={"marginBottom": "8px", "display": "flex", "alignItems": "center"})
            for f in subfolders
        ]
    except Exception as e:
        return html.Div(f"Error reading folders: {e}", style={"color": "#c00"})

# Callback to show confirmation dialog when bin is clicked
# Single global confirm dialog and store for delete
from dash.dependencies import ALL

# Show confirm dialog when any delete button is clicked

# Callback: perform deletion when dialog is confirmed
@dash.callback(
    Output("imported-runs-list", "children"),
    Output("wt-homologation-message-area", "children"),
    Input("current-homologation-store", "data"),
    Input({"type": "import-run-btn", "index": ALL}, "n_clicks"),
    Input("imported-runs-table", "active_cell"),
    Input("import-message-store", "data"),
    State("imported-runs-table", "data"),
    prevent_initial_call=False
)
def update_imported_runs_list(homologation, n_clicks_list, active_cell, last_message, table_data):
    import os
    ctx = callback_context
    message = last_message
    run_fields = []
    
    # Helper function to get map options from wt_maps folder
    def get_map_options(homologation_data):
        if not homologation_data or "wt_maps" not in homologation_data:
            return []
        wt_maps_path = homologation_data["wt_maps"]
        if not os.path.exists(wt_maps_path):
            return []
        try:
            json_files = [f for f in os.listdir(wt_maps_path) if f.endswith(".json")]
            # Remove .json extension for display
            return [f.replace(".json", "") for f in json_files]
        except Exception:
            return []

    # If no homologation data, still render an empty table with dropdown options from app config
    if not homologation:
        app_config_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "config", "run_plot_config.json")
        )
        app_config = {}
        if os.path.exists(app_config_path):
            with open(app_config_path, "r") as f:
                app_config = json.load(f)
        run_type_options = list(app_config.keys())
        map_options = get_map_options(homologation)
        run_fields = [
            dash_table.DataTable(
                id="imported-runs-table",
                columns=[
                    {"name": "Run", "id": "run", "editable": False},
                    {"name": "Description", "id": "description", "editable": True},
                    {"name": "Weighted Cz", "id": "weighted_Cz", "editable": False},
                    {"name": "Weighted Cx", "id": "weighted_Cx", "editable": False},
                    {"name": "Offset Cz", "id": "offset_Cz", "editable": True},
                    {"name": "Offset Cx", "id": "offset_Cx", "editable": True},
                    {"name": "Run Type", "id": "run_type", "presentation": "dropdown", "editable": True},
                    {"name": "Map", "id": "map", "presentation": "dropdown", "editable": True},
                    {"name": "Delete", "id": "delete", "presentation": "markdown", "editable": False}
                ],
                data=[],
                dropdown={
                    "run_type": {
                        "options": [{"label": opt, "value": opt} for opt in run_type_options],
                        "clearable": True
                    },
                    "map": {
                        "options": [{"label": opt, "value": opt} for opt in map_options],
                        "clearable": True
                    }
                },
                editable=True,
                style_table={"overflowX": "auto", "overflowY": "visible"},
                style_cell={
                    "textAlign": "left",
                    "minWidth": "60px",
                    "maxWidth": "220px",
                    "whiteSpace": "normal",
                    "fontSize": "12px",
                    "padding": "6px"
                },
                style_cell_conditional=[
                    {"if": {"column_id": "run"}, "width": "120px"},
                    {"if": {"column_id": "description"}, "width": "30%", "maxWidth": "320px"},
                    {"if": {"column_id": "weighted_Cz"}, "width": "90px", "textAlign": "center"},
                    {"if": {"column_id": "weighted_Cx"}, "width": "90px", "textAlign": "center"},
                    {"if": {"column_id": "offset_Cz"}, "width": "90px", "textAlign": "center"},
                    {"if": {"column_id": "offset_Cx"}, "width": "90px", "textAlign": "center"},
                    {"if": {"column_id": "run_type"}, "width": "160px"},
                    {"if": {"column_id": "map"}, "width": "160px"},
                    {"if": {"column_id": "delete"}, "width": "36px", "minWidth": "36px", "maxWidth": "36px", "textAlign": "center", "padding": "0", "overflow": "hidden"}
                ],
                css=[
                    {"selector": ".dash-cell.column-delete", "rule": "cursor: pointer; background: #ffeaea;"},
                    {"selector": ".dash-cell > div", "rule": "overflow: visible !important;"},
                    {"selector": ".Select-menu-outer", "rule": "display: block !important; z-index: 10000;"}
                ],
                style_header={"fontWeight": "bold", "backgroundColor": "#f5f5f5"},
                style_header_conditional=[
                    {"if": {"column_id": c}, "textAlign": "center"}
                    for c in ["weighted_Cz", "weighted_Cx", "offset_Cz", "offset_Cx", "delete"]
                ],
                page_size=20,
            )
        ]
        return run_fields, (message or "Load a homologation to see runs.")

    # Ignore non-delete cell clicks to allow inline editing without rerendering the table
    if ctx.triggered and len(ctx.triggered) == 1 and ctx.triggered[0]["prop_id"] == "imported-runs-table.active_cell":
        if not active_cell or active_cell.get("column_id") != "delete":
            return no_update, message

    # Handle delete icon click
    if active_cell and active_cell.get("column_id") == "delete":
        run_to_delete = table_data[active_cell["row"]]["run"]
        h5_path = homologation.get("h5_path") if homologation else None
        if h5_path and os.path.exists(h5_path):
            with h5py.File(h5_path, "a") as h5f:
                if "wt_runs" in h5f and run_to_delete in h5f["wt_runs"]:
                    del h5f["wt_runs"][run_to_delete]
                    message = f"Deleted run {run_to_delete}."
                else:
                    message = f"Run {run_to_delete} not found."
        else:
            message = "HDF5 file not found."
        active_cell = None  # Prevent repeat deletion if callback is triggered again
    # --- Load run_plot_config.json and set run_type_options at the top ---
    

    message = None
    # --- Load run_plot_config.json and set run_type_options at the top ---
    # Merge app-level config (defaults) with reference-folder config (overrides/extra)
    app_config_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "config", "run_plot_config.json")
    )
    app_config = {}
    if os.path.exists(app_config_path):
        with open(app_config_path, "r") as f:
            app_config = json.load(f)

    ref_config = {}
    if homologation and "reference_folder" in homologation:
        config_dst_dir = os.path.join(homologation["reference_folder"], "config")
        candidate = os.path.join(config_dst_dir, "run_plot_config.json")
        if os.path.exists(candidate):
            with open(candidate, "r") as f:
                ref_config = json.load(f)

    # Merge with reference taking precedence for duplicate keys
    run_plot_config = {**app_config, **ref_config}
    run_type_options = list(run_plot_config.keys())
    map_options = get_map_options(homologation)
    # --- END run_plot_config.json loading ---

    # Import logic
    if ctx.triggered and len(ctx.triggered) == 1 and "import-run-btn" in ctx.triggered[0]["prop_id"]:
        
        triggered = ctx.triggered[0]["prop_id"].split(".")[0]
        import_id = eval(triggered)
        folder_name = import_id["index"]
        # Step 1: Set 'Importing...' message
        importing_msg = f"Importing run {folder_name} ..."
        # Step 2: Try import
        if homologation:
            try:
                data_source_folder = homologation.get("data_source_folder")
                h5_path = homologation.get("h5_path")
                d1_path = os.path.join(data_source_folder, folder_name, "d1.asc")
                d2_path = os.path.join(data_source_folder, folder_name, "d2.asc")
                def load_d1_with_colnames(path):
                    with open(path) as f:
                        lines = f.readlines()
                    colnames = lines[3].strip().split('\t')
                    import io
                    data_lines = [line for i, line in enumerate(lines) if i > 4]
                    data = np.genfromtxt(io.StringIO(''.join(data_lines)), delimiter='\t')
                    return data, colnames
                d1, d1_colnames = load_d1_with_colnames(d1_path)
                d2, d2_colnames = load_d1_with_colnames(d2_path)
                # Also load a structured version of d1 to enable per-setpoint alignment when possible
                d1_structured, d1_structured_cols = load_d1_structured(d1_path)
                with h5py.File(h5_path, "a") as h5f:
                    if "wt_runs" not in h5f:
                        wt_runs = h5f.create_group("wt_runs")
                    else:
                        wt_runs = h5f["wt_runs"]
                    if folder_name in wt_runs:
                        del wt_runs[folder_name]
                    run_grp = wt_runs.create_group(folder_name)
                    d1_ds = run_grp.create_dataset("d1", data=d1)
                    d1_ds.attrs["columns"] = np.array(d1_colnames, dtype='S')
                    # Store conversion constants on datasets for downstream processing
                    d1_ds.attrs["PSF_to_Pa"] = PSF_TO_PA
                    d1_ds.attrs["LBF_to_Newtons"] = LBF_TO_NEWTONS

                    d2_ds = run_grp.create_dataset("d2", data=d2)
                    d2_ds.attrs["columns"] = np.array(d2_colnames, dtype='S')
                    d2_ds.attrs["PSF_to_Pa"] = PSF_TO_PA
                    d2_ds.attrs["LBF_to_Newtons"] = LBF_TO_NEWTONS
                    run_grp.attrs["description"] = "no description available"
                    # Don't initialize weighted values to 0.0 - they will be computed from d1_processed
                    # run_grp.attrs["weighted_Cz"] = 0.0
                    # run_grp.attrs["weighted_Cx"] = 0.0
                    run_grp.attrs["offset_Cz"] = 0.0
                    run_grp.attrs["offset_Cx"] = 0.0
                    # Store conversion constants at run level too
                    run_grp.attrs["PSF_to_Pa"] = PSF_TO_PA
                    run_grp.attrs["LBF_to_Newtons"] = LBF_TO_NEWTONS
                    # Set run_type to the first option from run_plot_config
                    run_grp.attrs["run_type"] = run_type_options[0] if run_type_options else ""
                    # Set map to the first option from wt_maps
                    selected_map = map_options[0] if map_options else ""
                    run_grp.attrs["map"] = selected_map
                    
                    # Create d1_processed dataset with setpoints from the selected map and d1 data
                    if selected_map:
                        # Validate row counts before creation
                        sp_data_for_count, _sp_cols = load_setpoints_from_map(selected_map)
                        map_rows = len(sp_data_for_count) if sp_data_for_count else 0
                        d1_rows = int(d1.shape[0]) if hasattr(d1, 'shape') and len(d1.shape) > 0 else 0
                        if map_rows > 0 and d1_rows > 0 and map_rows != d1_rows:
                            message = f"Error: map has {map_rows} rows but d1 has {d1_rows} rows. d1_processed not created."
                        else:
                            # Create combined data with map setpoints and d1 columns
                            combined_data, combined_columns = create_d1_processed_data(
                                selected_map, d1, d1_colnames, d1_structured, d1_structured_cols
                            )
                            if combined_data and combined_columns:
                                # Convert structured data to numpy array
                                combined_array = np.array(combined_data, dtype='S50')  # S50 for string up to 50 chars
                                d1_processed_ds = run_grp.create_dataset("d1_processed", data=combined_array)
                                d1_processed_ds.attrs["description"] = f"Setpoints from map: {selected_map} with d1 data columns"
                                # Store column names as attributes
                                d1_processed_ds.attrs["columns"] = np.array(combined_columns, dtype='S50')
                                # Compute weighted coefficients and store on run
                                ok, min_cd_msg = compute_weighted_coeffs_from_d1_processed(run_grp)
                                if not ok:
                                    message = (message + f"; {min_cd_msg}") if message else min_cd_msg
                                else:
                                    try:
                                        cz_val = float(run_grp.attrs.get("weighted_Cz", 0.0))
                                        cx_val = float(run_grp.attrs.get("weighted_Cx", 0.0))
                                        min_cd_val = float(run_grp.attrs.get("min_CD", 0.0))
                                        min_cd_sp = run_grp.attrs.get("min_CD_setpoint", "Unknown")
                                        if isinstance(min_cd_sp, bytes):
                                            min_cd_sp = min_cd_sp.decode()
                                        msg2 = f"Computed weighted Cz={cz_val:.4f}, Cx={cx_val:.4f} for run {folder_name}. {min_cd_msg}"
                                        message = (message + "; " + msg2) if message else msg2
                                    except Exception:
                                        pass
                            else:
                                # Create empty dataset if no setpoints found
                                empty_setpoints = np.array([], dtype='S50').reshape(0, 0)
                                d1_processed_ds = run_grp.create_dataset("d1_processed", data=empty_setpoints)
                                d1_processed_ds.attrs["description"] = f"No setpoints found for map: {selected_map}"
                                d1_processed_ds.attrs["columns"] = np.array([], dtype='S50')
                    else:
                        # Create empty dataset if no map selected
                        empty_setpoints = np.array([], dtype='S50').reshape(0, 0)
                        d1_processed_ds = run_grp.create_dataset("d1_processed", data=empty_setpoints)
                        d1_processed_ds.attrs["description"] = "No map selected"
                        d1_processed_ds.attrs["columns"] = np.array([], dtype='S50')

                message = f"Imported {folder_name} successfully."
            except Exception as e:
                message = f"Error: {e}"
        # Always reload the run list after import
        h5_path = homologation["h5_path"] if homologation and homologation.get("h5_path") else None
        run_fields = []
        if h5_path:
            exists = os.path.exists(h5_path)
            if not exists:
                pass
            try:
                with h5py.File(h5_path, "r") as h5f:
                    if "wt_runs" in h5f:
                        runs = list(h5f["wt_runs"].keys())
                        table_data = []
                        for run in runs:
                            run_attrs = h5f["wt_runs"][run].attrs
                            print(f"[DEBUG] Run '{run}' attributes: {dict(run_attrs)}")
                            def get_attr(key):
                                v = run_attrs.get(key, "")
                                if isinstance(v, bytes):
                                    v = v.decode()
                                return v
                            
                            def get_weighted_attr(key):
                                v = run_attrs.get(key, 0.0)
                                if isinstance(v, bytes):
                                    v = v.decode()
                                print(f"[DEBUG] get_weighted_attr({key}): raw_value={v}, type={type(v)}")
                                try:
                                    val = float(v)
                                    formatted = f"{val:.4f}"  # Changed to 4 decimal places
                                    print(f"[DEBUG] get_weighted_attr({key}): formatted={formatted}")
                                    return formatted
                                except (ValueError, TypeError) as e:
                                    print(f"[DEBUG] get_weighted_attr({key}): error={e}")
                                    return "0.0000"  # Changed to 4 decimal places
                            rt_value = get_attr("run_type")
                            # Ensure selected value is valid and present in the dropdown options
                            if (not rt_value) or (run_type_options and rt_value not in run_type_options):
                                rt_value = run_type_options[0] if run_type_options else ""
                            # Validate map value
                            map_value = get_attr("map")
                            if (not map_value) or (map_options and map_value not in map_options):
                                map_value = map_options[0] if map_options else ""
                            
                            table_data.append({
                                "run": run,
                                "description": get_attr("description"),
                                "weighted_Cz": get_weighted_attr("weighted_Cz"),
                                "weighted_Cx": get_weighted_attr("weighted_Cx"),
                                "offset_Cz": get_attr("offset_Cz"),
                                "offset_Cx": get_attr("offset_Cx"),
                                "run_type": rt_value,
                                "map": map_value,
                                "delete": "🗑️"
                            })
                        run_fields = [
                            dash_table.DataTable(
                                id="imported-runs-table",
                                columns=[
                                    {"name": "Run", "id": "run", "editable": False},
                                    {"name": "Description", "id": "description", "editable": True},
                                    {"name": "Weighted Cz", "id": "weighted_Cz", "editable": False},
                                    {"name": "Weighted Cx", "id": "weighted_Cx", "editable": False},
                                    {"name": "Offset Cz", "id": "offset_Cz", "editable": True},
                                    {"name": "Offset Cx", "id": "offset_Cx", "editable": True},
                                    {"name": "Run Type", "id": "run_type", "presentation": "dropdown", "editable": True},
                                    {"name": "Map", "id": "map", "presentation": "dropdown", "editable": True},
                                    {"name": "Delete", "id": "delete", "presentation": "markdown", "editable": False}
                                ],
                                data=table_data,
                                dropdown={
                                    "run_type": {
                                        "options": [{"label": opt, "value": opt} for opt in run_type_options],
                                        "clearable": True
                                    },
                                    "map": {
                                        "options": [{"label": opt, "value": opt} for opt in map_options],
                                        "clearable": True
                                    }
                                },
                                editable=True,
                                style_table={"overflowX": "auto", "overflowY": "visible"},
                                style_cell={
                                    "textAlign": "left",
                                    "minWidth": "60px",
                                    "maxWidth": "220px",
                                    "whiteSpace": "normal",
                                    "fontSize": "12px",
                                    "padding": "6px"
                                },
                                style_cell_conditional=[
                                    {"if": {"column_id": "run"}, "width": "120px"},
                                    {"if": {"column_id": "description"}, "width": "30%", "maxWidth": "320px"},
                                    {"if": {"column_id": "weighted_Cz"}, "width": "90px", "textAlign": "center"},
                                    {"if": {"column_id": "weighted_Cx"}, "width": "90px", "textAlign": "center"},
                                    {"if": {"column_id": "offset_Cz"}, "width": "90px", "textAlign": "center"},
                                    {"if": {"column_id": "offset_Cx"}, "width": "90px", "textAlign": "center"},
                                    {"if": {"column_id": "run_type"}, "width": "160px"},
                                    {"if": {"column_id": "map"}, "width": "160px"},
                                    {"if": {"column_id": "delete"}, "width": "36px", "minWidth": "36px", "maxWidth": "36px", "textAlign": "center", "padding": "0", "overflow": "hidden"}
                                ],
                                css=[
                                    {"selector": ".dash-cell.column-delete", "rule": "cursor: pointer; background: #ffeaea;"},
                                    {"selector": ".dash-cell > div", "rule": "overflow: visible !important;"},
                                    {"selector": ".Select-menu-outer", "rule": "display: block !important; z-index: 10000;"}
                                ],
                                style_header={"fontWeight": "bold", "backgroundColor": "#f5f5f5"},
                                style_header_conditional=[
                                    {"if": {"column_id": c}, "textAlign": "center"}
                                    for c in ["weighted_Cz", "weighted_Cx", "offset_Cz", "offset_Cx", "delete"]
                                ],
                                page_size=20,
                            )
                        ]

            except Exception as e:
                message = f"Error: {e}"
        # Show success/error message, else show importing message
        return run_fields, (message if message else importing_msg)

    # Not triggered by import icon: just update run list, keep previous message
    h5_path = homologation["h5_path"] if homologation and homologation.get("h5_path") else None
    run_fields = []
    if h5_path:
        import os
        exists = os.path.exists(h5_path)
        if not exists:
            pass
        try:
            with h5py.File(h5_path, "r") as h5f:
                if "wt_runs" in h5f:
                    runs = list(h5f["wt_runs"].keys())
                    
                    table_data = []
                    for run in runs:
                        run_attrs = h5f["wt_runs"][run].attrs
                        print(f"[DEBUG] Run '{run}' attributes: {dict(run_attrs)}")
                        def get_attr(key):
                            v = run_attrs.get(key, "")
                            if isinstance(v, bytes):
                                v = v.decode()
                            return v
                        
                        def get_weighted_attr(key):
                            v = run_attrs.get(key, 0.0)
                            if isinstance(v, bytes):
                                v = v.decode()
                            print(f"[DEBUG] get_weighted_attr({key}): raw_value={v}, type={type(v)}")
                            try:
                                val = float(v)
                                formatted = f"{val:.4f}"  # Changed to 4 decimal places
                                print(f"[DEBUG] get_weighted_attr({key}): formatted={formatted}")
                                return formatted
                            except (ValueError, TypeError) as e:
                                print(f"[DEBUG] get_weighted_attr({key}): error={e}")
                                return "0.0000"  # Changed to 4 decimal places
                        
                        def to_plain_type(val):
                            import numpy as np
                            # Convert numpy types to Python types
                            if isinstance(val, (np.generic, np.ndarray)):
                                return val.item()
                            # Remove extra quotes from strings
                            if isinstance(val, str):
                                v = val.strip()
                                if v.startswith("'") and v.endswith("'") and len(v) > 1:
                                    v = v[1:-1]
                                # Try to convert numeric strings to float or int
                                try:
                                    if '.' in v or 'e' in v.lower():
                                        return float(v)
                                    else:
                                        return int(v)
                                except (ValueError, TypeError):
                                    return v
                                return v
                            return val
                        # Validate run_type value
                        rt_value = to_plain_type(get_attr("run_type"))
                        if (not rt_value) or (run_type_options and rt_value not in run_type_options):
                            rt_value = run_type_options[0] if run_type_options else ""
                        
                        # Validate map value
                        map_value = to_plain_type(get_attr("map"))
                        if (not map_value) or (map_options and map_value not in map_options):
                            map_value = map_options[0] if map_options else ""

                        table_data.append({
                            "run": to_plain_type(run),
                            "description": to_plain_type(get_attr("description")),
                            "weighted_Cz": get_weighted_attr("weighted_Cz"),
                            "weighted_Cx": get_weighted_attr("weighted_Cx"),
                            "offset_Cz": to_plain_type(get_attr("offset_Cz")),
                            "offset_Cx": to_plain_type(get_attr("offset_Cx")),
                            "run_type": rt_value,
                            "map": map_value,
                            "delete": "🗑️",
                        })

                    run_fields = [
                        dash_table.DataTable(
                            id="imported-runs-table",
                            columns=[
                                {"name": "Run", "id": "run", "editable": False},
                                {"name": "Description", "id": "description", "editable": True},
                                {"name": "Weighted Cz", "id": "weighted_Cz", "editable": False},
                                {"name": "Weighted Cx", "id": "weighted_Cx", "editable": False},
                                {"name": "Offset Cz", "id": "offset_Cz", "editable": True},
                                {"name": "Offset Cx", "id": "offset_Cx", "editable": True},
                                {"name": "Run Type", "id": "run_type", "presentation": "dropdown", "editable": True},
                                {"name": "Map", "id": "map", "presentation": "dropdown", "editable": True},
                                {"name": "Delete", "id": "delete", "presentation": "markdown", "editable": False}
                            ],
                            data=table_data,
                            dropdown={
                                "run_type": {
                                    "options": [{"label": opt, "value": opt} for opt in run_type_options],
                                    "clearable": True
                                },
                                "map": {
                                    "options": [{"label": opt, "value": opt} for opt in map_options],
                                    "clearable": True
                                }
                            },
                            editable=True,
                            style_table={"overflowX": "auto", "overflowY": "visible"},
                            style_cell={
                                "textAlign": "left",
                                "minWidth": "60px",
                                "maxWidth": "220px",
                                "whiteSpace": "normal",
                                "fontSize": "12px",
                                "padding": "6px"
                            },
                            style_cell_conditional=[
                                {"if": {"column_id": "run"}, "width": "120px"},
                                {"if": {"column_id": "description"}, "width": "30%", "maxWidth": "320px"},
                                {"if": {"column_id": "weighted_Cz"}, "width": "90px", "textAlign": "center"},
                                {"if": {"column_id": "weighted_Cx"}, "width": "90px", "textAlign": "center"},
                                {"if": {"column_id": "offset_Cz"}, "width": "90px", "textAlign": "center"},
                                {"if": {"column_id": "offset_Cx"}, "width": "90px", "textAlign": "center"},
                                {"if": {"column_id": "run_type"}, "width": "160px"},
                                {"if": {"column_id": "map"}, "width": "160px"},
                                {"if": {"column_id": "delete"}, "width": "36px", "minWidth": "36px", "maxWidth": "36px", "textAlign": "center", "padding": "0", "overflow": "hidden"}
                            ],
                            css=[
                                {"selector": ".dash-cell.column-delete", "rule": "cursor: pointer; background: #ffeaea;"},
                                {"selector": ".dash-cell > div", "rule": "overflow: visible !important;"},
                                {"selector": ".Select-menu-outer", "rule": "display: block !important; z-index: 10000;"}
                            ],
                            style_header={"fontWeight": "bold", "backgroundColor": "#f5f5f5"},
                            style_header_conditional=[
                                {"if": {"column_id": c}, "textAlign": "center"}
                                for c in ["weighted_Cz", "weighted_Cx", "offset_Cz", "offset_Cx", "delete"]
                            ],
                            page_size=20,
                        )
                    ]


        except Exception as e:
            last_message = f"Error: {e}"
 
    return run_fields, (last_message if last_message else "")


# Callback to handle table data changes and save to HDF5
@dash.callback(
    Output("import-message-store", "data"),
    Input("imported-runs-table", "data"),
    State("current-homologation-store", "data"),
    prevent_initial_call=True
)
def save_table_changes(table_data, homologation):
    if not table_data or not homologation:
        return no_update
    
    h5_path = homologation.get("h5_path")
    if not h5_path or not os.path.exists(h5_path):
        return "HDF5 file not found"
    
    try:
        with h5py.File(h5_path, "a") as h5f:
            if "wt_runs" not in h5f:
                return "No runs found in HDF5 file"
            
            for row in table_data:
                run_name = row.get("run")
                if run_name and run_name in h5f["wt_runs"]:
                    run_grp = h5f["wt_runs"][run_name]
                    
                    # Update attributes from table data
                    run_grp.attrs["description"] = str(row.get("description", ""))
                    run_grp.attrs["weighted_Cz"] = float(row.get("weighted_Cz", 0.0))
                    run_grp.attrs["weighted_Cx"] = float(row.get("weighted_Cx", 0.0))
                    run_grp.attrs["offset_Cz"] = float(row.get("offset_Cz", 0.0))
                    run_grp.attrs["offset_Cx"] = float(row.get("offset_Cx", 0.0))
                    run_grp.attrs["run_type"] = str(row.get("run_type", ""))
                    
                    # Handle map change and update d1_processed dataset
                    new_map = str(row.get("map", ""))
                    old_map = run_grp.attrs.get("map", "")
                    if isinstance(old_map, bytes):
                        old_map = old_map.decode()
                    
                    # Update d1_processed dataset if map has changed
                    if new_map != old_map:
                        # Always remove existing d1_processed dataset if it exists
                        if "d1_processed" in run_grp:
                            del run_grp["d1_processed"]

                        # If a new map is provided, validate and recreate
                        if new_map:
                            # Get d1 data and columns from the existing dataset
                            d1_data = []
                            d1_columns = []
                            d1_structured = None
                            d1_structured_cols = []
                            if "d1" in run_grp:
                                d1_dataset = run_grp["d1"]
                                d1_data = d1_dataset[:]
                                if "columns" in d1_dataset.attrs:
                                    d1_columns = [col.decode() if isinstance(col, bytes) else col for col in d1_dataset.attrs["columns"]]
                                # Attempt to load structured d1 via file path if possible is not trivial here; use raw arrays

                            # Validate row counts between selected map and d1
                            sp_data_for_count, _sp_cols = load_setpoints_from_map(new_map)
                            map_rows = len(sp_data_for_count) if sp_data_for_count else 0
                            d1_rows = int(d1_data.shape[0]) if hasattr(d1_data, 'shape') and len(d1_data.shape) > 0 else (len(d1_data) if isinstance(d1_data, (list, np.ndarray)) else 0)
                            if map_rows > 0 and d1_rows > 0 and map_rows != d1_rows:
                                # Restore d1_processed with old map before returning error
                                if old_map:
                                    combined_data, combined_columns = create_d1_processed_data(old_map, d1_data, d1_columns, d1_structured, d1_structured_cols)
                                    if combined_data and combined_columns:
                                        combined_array = np.array(combined_data, dtype='S50')
                                        d1_processed_ds = run_grp.create_dataset("d1_processed", data=combined_array)
                                        d1_processed_ds.attrs["description"] = f"Setpoints from map: {old_map} with d1 data columns"
                                        d1_processed_ds.attrs["columns"] = np.array(combined_columns, dtype='S50')
                                        compute_weighted_coeffs_from_d1_processed(run_grp)
                                return f"Error: map has {map_rows} rows but d1 has {d1_rows} rows. Map remains: {old_map}"

                            # Create combined data with map setpoints and d1 columns
                            combined_data, combined_columns = create_d1_processed_data(new_map, d1_data, d1_columns, d1_structured, d1_structured_cols)
                            if combined_data and combined_columns:
                                combined_array = np.array(combined_data, dtype='S50')
                                d1_processed_ds = run_grp.create_dataset("d1_processed", data=combined_array)
                                d1_processed_ds.attrs["description"] = f"Setpoints from map: {new_map} with d1 data columns"
                                d1_processed_ds.attrs["columns"] = np.array(combined_columns, dtype='S50')
                                ok, min_cd_msg = compute_weighted_coeffs_from_d1_processed(run_grp)
                                if not ok:
                                    return min_cd_msg
                                else:
                                    # Only update map attribute after successful creation
                                    run_grp.attrs["map"] = new_map
                                    try:
                                        cz_val = float(run_grp.attrs.get("weighted_Cz", 0.0))
                                        cx_val = float(run_grp.attrs.get("weighted_Cx", 0.0))
                                        min_cd_val = float(run_grp.attrs.get("min_CD", 0.0))
                                        min_cd_sp = run_grp.attrs.get("min_CD_setpoint", "Unknown")
                                        if isinstance(min_cd_sp, bytes):
                                            min_cd_sp = min_cd_sp.decode()
                                        return f"Computed weighted Cz={cz_val:.4f}, Cx={cx_val:.4f} for run {run_name if 'run_name' in locals() else ''}. {min_cd_msg}"
                                    except Exception:
                                        return "Computed weighted coefficients"
                            else:
                                # If there is no setpoint data, do not recreate
                                return f"Error: No setpoints found for map: {new_map}. Map remains: {old_map}"
                        else:
                            # No map selected; clear map and do not recreate
                            run_grp.attrs["map"] = new_map
                            return "No map selected; d1_processed not created"
        
        return "Changes saved successfully"
    except Exception as e:
        return f"Error saving changes: {e}"

 
