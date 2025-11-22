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

def compute_weighted_coeffs_from_d1_processed(run_grp, map_name=None, min_Cx_weighting=20, instability=False, speed_correction=None):
    """Compute weighted_Cz and weighted_Cx based on d1_processed dataset and store as run attributes.

    Args:
        run_grp: HDF5 group containing the run data
        map_name: Name of the map being used (for logging)
        min_Cx_weighting: Minimum Cx weighting percentage (default 20, meaning 20% or 0.2)
        instability: Boolean flag to enable speed-matched tares (default False)
        speed_correction: Dict with speed correction config (None if not specified)

    Given:
    - drag_tare = D value at appropriate Zero setpoint (speed-matched if instability=True)
    - lift_tare = L value at appropriate Zero setpoint (speed-matched if instability=True)
    - CL = (L - lift_tare) * LBF_TO_NEWTONS / (DYNPR * PSF_TO_PA)
    - CLW = CL * w_cz (only for valid measurements)
      weighted_Cz = sum(CLW) / sum(ALL w_cz from map)
    - CD = (D - drag_tare) * LBF_TO_NEWTONS / (DYNPR * PSF_TO_PA)
    - min_CD = minimum CD from rows with valid (non-NaN) w_cx
    - CDW = CD * w_cx (only for valid measurements)
    
    Weighted Cx calculation (all maps):
    - weighted_Cx = sum(CDW) / 100 + (min_Cx_weighting / 100) * min_CD
    
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
        road_speed_idx = _find_col_index('road_speed', cols)
        wind_speed_idx = _find_col_index('wind_speed', cols)

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

        # Find tares - support instability mode with speed-matched zeros
        # Build tare dictionaries: by speed and by name
        tare_values = {}  # {speed: {'D': value, 'L': value}}
        tare_by_name = {}  # {setpoint_name: {'D': value, 'L': value}}
        for r in range(data.shape[0]):
            sp_val = data[r, sp_idx]
            if isinstance(sp_val, (bytes, bytearray)):
                sp_val = sp_val.decode()
            sp_name = str(sp_val)
            if 'Zero' in sp_name:
                speed = to_float(data[r, road_speed_idx]) if road_speed_idx != -1 else 60.0
                if speed not in tare_values:
                    tare_values[speed] = {}
                if sp_name not in tare_by_name:
                    tare_by_name[sp_name] = {}
                if D_idx != -1:
                    tare_values[speed]['D'] = to_float(data[r, D_idx])
                    tare_by_name[sp_name]['D'] = to_float(data[r, D_idx])
                if L_idx != -1:
                    tare_values[speed]['L'] = to_float(data[r, L_idx])
                    tare_by_name[sp_name]['L'] = to_float(data[r, L_idx])
        
        # Helper function to get appropriate tare value based on speed
        def get_tare_for_speed(speed, tare_type):
            # Lift tare ALWAYS uses ZeroL (2 mph), regardless of instability mode
            if tare_type == 'L':
                return tare_by_name.get('ZeroL', {}).get('L', 0.0)
            
            # Drag tare handling
            if not instability:
                # Non-instability mode: use ZeroD1 specifically by name
                return tare_by_name.get('ZeroD1', {}).get('D', 0.0)
            else:
                # Instability mode: match speed for drag (50 mph -> ZeroD2, 60 mph -> ZeroD1)
                if speed in tare_values and 'D' in tare_values[speed]:
                    return tare_values[speed]['D']
                # Fallback to closest speed
                speeds = sorted([s for s in tare_values.keys() if 'D' in tare_values[s]])
                if not speeds:
                    return 0.0
                closest_speed = min(speeds, key=lambda s: abs(s - speed))
                return tare_values[closest_speed].get('D', 0.0)
        
        if instability:
            print(f"[WT] Instability mode enabled for compute. Found tare values at speeds: {sorted(tare_values.keys())}")
        
        # Calculate speed correction deltas if configured
        delta_CL = 0.0
        delta_CD = 0.0
        unstable_50_name = None
        if speed_correction and instability:
            ref_60_name = speed_correction.get('reference_60')
            ref_50_name = speed_correction.get('reference_50')
            unstable_50_name = speed_correction.get('unstable_50')
            
            if ref_60_name and ref_50_name and unstable_50_name:
                # Find the data for reference setpoints
                ref_60_L, ref_60_D, ref_60_dynpr, ref_60_speed = None, None, None, None
                ref_50_L, ref_50_D, ref_50_dynpr, ref_50_speed = None, None, None, None
                
                for r in range(data.shape[0]):
                    sp_val = data[r, sp_idx]
                    if isinstance(sp_val, (bytes, bytearray)):
                        sp_val = sp_val.decode()
                    sp_name_check = str(sp_val)
                    
                    if sp_name_check == ref_60_name:
                        ref_60_L = to_float(data[r, L_idx])
                        ref_60_D = to_float(data[r, D_idx]) if D_idx != -1 else float('nan')
                        ref_60_dynpr = to_float(data[r, DYNPR_idx])
                        ref_60_speed = to_float(data[r, road_speed_idx]) if road_speed_idx != -1 else 60.0
                    elif sp_name_check == ref_50_name:
                        ref_50_L = to_float(data[r, L_idx])
                        ref_50_D = to_float(data[r, D_idx]) if D_idx != -1 else float('nan')
                        ref_50_dynpr = to_float(data[r, DYNPR_idx])
                        ref_50_speed = to_float(data[r, road_speed_idx]) if road_speed_idx != -1 else 50.0
                
                # Calculate deltas if both reference points found
                if all(v is not None for v in [ref_60_L, ref_60_D, ref_60_dynpr, ref_60_speed, ref_50_L, ref_50_D, ref_50_dynpr, ref_50_speed]):
                    # Calculate CL and CD for reference at 60 mph
                    lift_tare_60 = get_tare_for_speed(ref_60_speed, 'L')
                    drag_tare_60 = get_tare_for_speed(ref_60_speed, 'D')
                    
                    CL_60 = 0.0
                    CD_60 = 0.0
                    if not (np.isnan(ref_60_L) or np.isnan(lift_tare_60) or np.isnan(ref_60_dynpr) or ref_60_dynpr == 0):
                        CL_60 = (ref_60_L - lift_tare_60) * LBF_TO_NEWTONS / (ref_60_dynpr * PSF_TO_PA)
                    if not (np.isnan(ref_60_D) or np.isnan(drag_tare_60) or np.isnan(ref_60_dynpr) or ref_60_dynpr == 0):
                        CD_60 = (ref_60_D - drag_tare_60) * LBF_TO_NEWTONS / (ref_60_dynpr * PSF_TO_PA)
                    
                    # Calculate CL and CD for reference at 50 mph
                    lift_tare_50 = get_tare_for_speed(ref_50_speed, 'L')
                    drag_tare_50 = get_tare_for_speed(ref_50_speed, 'D')
                    
                    CL_50 = 0.0
                    CD_50 = 0.0
                    if not (np.isnan(ref_50_L) or np.isnan(lift_tare_50) or np.isnan(ref_50_dynpr) or ref_50_dynpr == 0):
                        CL_50 = (ref_50_L - lift_tare_50) * LBF_TO_NEWTONS / (ref_50_dynpr * PSF_TO_PA)
                    if not (np.isnan(ref_50_D) or np.isnan(drag_tare_50) or np.isnan(ref_50_dynpr) or ref_50_dynpr == 0):
                        CD_50 = (ref_50_D - drag_tare_50) * LBF_TO_NEWTONS / (ref_50_dynpr * PSF_TO_PA)
                    
                    # Calculate deltas
                    delta_CL = CL_60 - CL_50
                    delta_CD = CD_60 - CD_50
                    
                    print(f"[WT] Speed correction (compute): {ref_60_name}@60mph CL={CL_60:.6f} CD={CD_60:.6f}, {ref_50_name}@50mph CL={CL_50:.6f} CD={CD_50:.6f}")
                    print(f"[WT] Speed correction deltas (compute): delta_CL={delta_CL:.6f}, delta_CD={delta_CD:.6f}")
                    print(f"[WT] Will apply correction to unstable setpoint: {unstable_50_name}")
        
        # First pass: find minimum CD from rows with valid w_cx (post speed-correction)
        min_cd = float('inf')
        min_cd_setpoint = "Unknown"
        valid_cd_values = []
        for r in range(data.shape[0]):
            # Get setpoint name
            sp_val = data[r, sp_idx]
            if isinstance(sp_val, (bytes, bytearray)):
                sp_val = sp_val.decode()
            sp_name_str = str(sp_val)
            
            # Exclude reference setpoints from min_CD calculation if speed_correction is configured
            if speed_correction and instability:
                ref_60_name = speed_correction.get('reference_60')
                ref_50_name = speed_correction.get('reference_50')
                if sp_name_str in [ref_60_name, ref_50_name]:
                    continue  # Skip reference setpoints
            
            D = to_float(data[r, D_idx]) if D_idx != -1 else float('nan')
            dynpr = to_float(data[r, DYNPR_idx])
            wcx = to_float(data[r, wcx_idx]) if wcx_idx != -1 else float('nan')
            road_speed = to_float(data[r, road_speed_idx]) if road_speed_idx != -1 else 60.0
            wind_speed = to_float(data[r, wind_speed_idx]) if wind_speed_idx != -1 else 60.0
            
            # Skip rows with wind_speed=0 (Zero/tare setpoints)
            if wind_speed == 0:
                continue
            
            # Get appropriate drag tare for this speed
            drag_tare = get_tare_for_speed(road_speed, 'D')
            
            if not (np.isnan(D) or np.isnan(drag_tare) or np.isnan(dynpr) or np.isnan(wcx) or dynpr == 0):
                cd = (D - drag_tare) * LBF_TO_NEWTONS / (dynpr * PSF_TO_PA)
                
                # Apply speed correction if this is the unstable setpoint
                if speed_correction and instability and unstable_50_name and sp_name_str == unstable_50_name:
                    if not np.isnan(cd):
                        cd = cd + delta_CD
                
                if not np.isnan(cd):
                    valid_cd_values.append(cd)
                    if cd < min_cd:
                        min_cd = cd
                        min_cd_setpoint = sp_name_str
        
        # Handle case where no valid CD values found
        if min_cd == float('inf'):
            min_cd = 0.0
            min_cd_setpoint = "None"
        
        print(f"[WT] Minimum CD found (valid w_cx, wind_speed>0, post-correction, excl. references): {min_cd:.6f} at setpoint '{min_cd_setpoint}'")
        
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
            road_speed = to_float(data[r, road_speed_idx]) if road_speed_idx != -1 else 60.0
            
            # Get appropriate tares for this speed
            lift_tare = get_tare_for_speed(road_speed, 'L')
            drag_tare = get_tare_for_speed(road_speed, 'D')
            
            # Accumulate weight sums for ALL valid weights (not just when measurements are valid)
            if not np.isnan(wcz):
                wcz_sum += wcz
            if not np.isnan(wcx):
                wcx_sum += wcx
            
            # Initialize for debug
            cl = float('nan')
            cd = float('nan')
            clw = float('nan')
            cdw = float('nan')

            # Calculate CL (Coefficient of Lift) first
            if not (np.isnan(L) or np.isnan(lift_tare) or np.isnan(dynpr) or dynpr == 0):
                cl = (L - lift_tare) * LBF_TO_NEWTONS / (dynpr * PSF_TO_PA)
            
            # Calculate CD (Coefficient of Drag) first
            if not (np.isnan(D) or np.isnan(drag_tare) or np.isnan(dynpr) or dynpr == 0):
                cd = (D - drag_tare) * LBF_TO_NEWTONS / (dynpr * PSF_TO_PA)
            
            # Apply speed correction if this is the unstable setpoint
            if speed_correction and instability and unstable_50_name:
                sp_val = data[r, sp_idx]
                if isinstance(sp_val, (bytes, bytearray)):
                    sp_val = sp_val.decode()
                if str(sp_val) == unstable_50_name:
                    cl_before = cl
                    cd_before = cd
                    if not np.isnan(cl):
                        cl = cl + delta_CL
                    if not np.isnan(cd):
                        cd = cd + delta_CD
                    print(f"[WT] Applied speed correction to {unstable_50_name}: CL {cl_before:.6f} -> {cl:.6f}, CD {cd_before:.6f} -> {cd:.6f}")
            
            # Calculate weighted values
            if not np.isnan(cl) and not np.isnan(wcz):
                clw = cl * wcz  # CLW = CL * w_cz
                clw_sum += clw
            
            if not np.isnan(cd) and not np.isnan(wcx):
                cdw = cd * wcx  # CDW = CD * w_cx (back to original formula)
                cdw_sum += cdw

            # Per-row debug for short map
            if map_name == "short":
                try:
                    print(f"[WT][short][weighted pass] r={r} CL={cl:.6f} w_cz={wcz} CLW={clw if not np.isnan(clw) else 'nan'} | CD={cd:.6f} w_cx={wcx} CDW={cdw if not np.isnan(cdw) else 'nan'}")
                except Exception:
                    pass
        
        # Final weighted values are normalized by the sum of weights
        weighted_cz = float(clw_sum / wcz_sum) if (wcz_sum != 0.0 and not np.isnan(wcz_sum)) else 0.0
        weighted_cx_base = float(cdw_sum / 100.0)  # Normalize by 100 instead of wcx_sum
        
        # Apply minimum CD offset using the min_Cx_weighting from the map (convert percentage to decimal)
        min_cx_offset = (min_Cx_weighting / 100.0) * min_cd
        weighted_cx = weighted_cx_base + min_cx_offset
        
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
        map_label = f" (map: {map_name})" if map_name else ""
        print(f"[WT] Run '{run_name}'{map_label}: weighted_Cz={weighted_cz:.6f}, weighted_Cx_base={weighted_cx_base:.6f}, min_CD_offset={min_cx_offset:.6f} ({min_Cx_weighting}%), weighted_Cx={weighted_cx:.6f}")
        
        # Return success message with minimum CD info
        success_msg = f"Minimum CD: {min_cd:.6f} at ride height '{min_cd_setpoint}'"
        return True, success_msg
    except Exception as e:
        return False, f"Error computing weighted coeffs: {e}"

def load_setpoints_from_map(map_name):
    """Load full setpoint data from the specified map in maps.json
    
    Returns:
        tuple: (structured_data, columns, min_Cx_weighting, instability, speed_correction)
            - structured_data: list of rows with setpoint data
            - columns: list of column names
            - min_Cx_weighting: minimum Cx weighting percentage (default 20 if not specified)
            - instability: boolean flag indicating if map has instability (default False)
            - speed_correction: dict with speed correction config (None if not specified)
    """
    try:
        maps_config_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "config", "maps.json")
        )
        if not os.path.exists(maps_config_path):
            return [], [], 20, False, None
        
        with open(maps_config_path, "r") as f:
            maps_data = json.load(f)
        
        if map_name in maps_data:
            map_config = maps_data[map_name]
            if not map_config:
                return [], [], 20, False, None
            
            # Handle new structure with "setpoints" and "min_Cx_weighting" keys
            if isinstance(map_config, dict) and "setpoints" in map_config:
                setpoints_data = map_config["setpoints"]
                min_Cx_weighting = map_config.get("min_Cx_weighting", 20)
                instability = map_config.get("instability", False)
                speed_correction = map_config.get("speed_correction", None)
            else:
                # Legacy structure: map_config is directly the setpoints array
                setpoints_data = map_config
                min_Cx_weighting = 20
                instability = False
                speed_correction = None
            
            if not setpoints_data:
                return [], [], min_Cx_weighting, instability, speed_correction
            
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
            
            return structured_data, columns, min_Cx_weighting, instability, speed_correction
        else:
            return [], [], 20, False, None
    except Exception as e:
        print(f"Error loading setpoints from map {map_name}: {e}")
        return [], [], 20, False, None

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
        setpoints_data, map_columns, _, instability, speed_correction = load_setpoints_from_map(map_name)
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

        # Helper function to safely convert to float
        def to_float(x):
            try:
                if isinstance(x, (bytes, bytearray)):
                    x = x.decode()
                return float(x)
            except Exception:
                return float('nan')
        
        # Find tare values - support instability mode with speed-matched zeros
        # Build tare dictionaries: by speed and by name
        tare_values = {}  # {speed: {'D': value, 'L': value}}
        tare_by_name = {}  # {setpoint_name: {'D': value, 'L': value}}
        
        if per_setpoint_values:
            # Collect all zero setpoints and their speeds
            for sp_name, vals in per_setpoint_values.items():
                if 'Zero' in sp_name:
                    # Find the speed for this zero setpoint from map
                    sp_col_idx = _find_col_index('setpoint', map_columns)
                    road_speed_idx = _find_col_index('road_speed', map_columns)
                    if sp_col_idx != -1 and road_speed_idx != -1:
                        speed = None
                        for sp_row in setpoints_data:
                            if str(sp_row[sp_col_idx]) == sp_name:
                                speed = to_float(sp_row[road_speed_idx])
                                break
                        if speed is not None:
                            if speed not in tare_values:
                                tare_values[speed] = {}
                            if sp_name not in tare_by_name:
                                tare_by_name[sp_name] = {}
                            if 'D' in vals:
                                tare_values[speed]['D'] = vals['D']
                                tare_by_name[sp_name]['D'] = vals['D']
                            if 'L' in vals:
                                tare_values[speed]['L'] = vals['L']
                                tare_by_name[sp_name]['L'] = vals['L']
        
        # Helper function to get appropriate tare value based on speed
        def get_tare_for_speed(speed, tare_type):
            # Lift tare ALWAYS uses ZeroL (2 mph), regardless of instability mode
            if tare_type == 'L':
                return tare_by_name.get('ZeroL', {}).get('L', 0.0)
            
            # Drag tare handling
            if not instability:
                # Non-instability mode: use ZeroD1 specifically by name
                return tare_by_name.get('ZeroD1', {}).get('D', 0.0)
            else:
                # Instability mode: match speed for drag (50 mph -> ZeroD2, 60 mph -> ZeroD1)
                if speed in tare_values and 'D' in tare_values[speed]:
                    return tare_values[speed]['D']
                # Fallback to closest speed
                speeds = sorted([s for s in tare_values.keys() if 'D' in tare_values[s]])
                if not speeds:
                    return 0.0
                closest_speed = min(speeds, key=lambda s: abs(s - speed))
                return tare_values[closest_speed].get('D', 0.0)
        
        if instability:
            print(f"[WT] Instability mode enabled for map '{map_name}'. Found tare values at speeds: {sorted(tare_values.keys())}")

        # Calculate speed correction deltas if configured (BEFORE min_CD calculation)
        delta_CL = 0.0
        delta_CD = 0.0
        unstable_50_name = None
        if speed_correction and instability:
            ref_60_name = speed_correction.get('reference_60')
            ref_50_name = speed_correction.get('reference_50')
            unstable_50_name = speed_correction.get('unstable_50')
            
            if ref_60_name and ref_50_name and unstable_50_name:
                # Find the data for reference setpoints
                ref_60_data = None
                ref_50_data = None
                
                sp_col_idx = _find_col_index('setpoint', map_columns)
                if sp_col_idx != -1:
                    for i, sp_row in enumerate(setpoints_data):
                        sp_name_check = str(sp_row[sp_col_idx])
                        if sp_name_check == ref_60_name:
                            # Get L, D, DYNPR for reference at 60 mph
                            if ref_60_name in per_setpoint_values:
                                ref_60_data = per_setpoint_values[ref_60_name]
                            elif i < len(d1_data) and 'L' in d1_column_indices and 'D' in d1_column_indices and 'DYNPR' in d1_column_indices:
                                ref_60_data = {
                                    'L': to_float(d1_data[i][d1_column_indices['L']]),
                                    'D': to_float(d1_data[i][d1_column_indices['D']]),
                                    'DYNPR': to_float(d1_data[i][d1_column_indices['DYNPR']]),
                                    'road_speed': to_float(sp_row[_find_col_index('road_speed', map_columns)])
                                }
                        elif sp_name_check == ref_50_name:
                            # Get L, D, DYNPR for reference at 50 mph
                            if ref_50_name in per_setpoint_values:
                                ref_50_data = per_setpoint_values[ref_50_name]
                            elif i < len(d1_data) and 'L' in d1_column_indices and 'D' in d1_column_indices and 'DYNPR' in d1_column_indices:
                                ref_50_data = {
                                    'L': to_float(d1_data[i][d1_column_indices['L']]),
                                    'D': to_float(d1_data[i][d1_column_indices['D']]),
                                    'DYNPR': to_float(d1_data[i][d1_column_indices['DYNPR']]),
                                    'road_speed': to_float(sp_row[_find_col_index('road_speed', map_columns)])
                                }
                
                # Calculate deltas if both reference points found
                if ref_60_data and ref_50_data:
                    # Calculate CL and CD for reference at 60 mph
                    L_60 = ref_60_data.get('L', 0.0)
                    D_60 = ref_60_data.get('D', 0.0)
                    dynpr_60 = ref_60_data.get('DYNPR', 0.0)
                    speed_60 = ref_60_data.get('road_speed', 60.0)
                    lift_tare_60 = get_tare_for_speed(speed_60, 'L')
                    drag_tare_60 = get_tare_for_speed(speed_60, 'D')
                    
                    CL_60 = 0.0
                    CD_60 = 0.0
                    if not (np.isnan(L_60) or np.isnan(lift_tare_60) or np.isnan(dynpr_60) or dynpr_60 == 0):
                        CL_60 = (L_60 - lift_tare_60) * LBF_TO_NEWTONS / (dynpr_60 * PSF_TO_PA)
                    if not (np.isnan(D_60) or np.isnan(drag_tare_60) or np.isnan(dynpr_60) or dynpr_60 == 0):
                        CD_60 = (D_60 - drag_tare_60) * LBF_TO_NEWTONS / (dynpr_60 * PSF_TO_PA)
                    
                    # Calculate CL and CD for reference at 50 mph
                    L_50 = ref_50_data.get('L', 0.0)
                    D_50 = ref_50_data.get('D', 0.0)
                    dynpr_50 = ref_50_data.get('DYNPR', 0.0)
                    speed_50 = ref_50_data.get('road_speed', 50.0)
                    lift_tare_50 = get_tare_for_speed(speed_50, 'L')
                    drag_tare_50 = get_tare_for_speed(speed_50, 'D')
                    
                    CL_50 = 0.0
                    CD_50 = 0.0
                    if not (np.isnan(L_50) or np.isnan(lift_tare_50) or np.isnan(dynpr_50) or dynpr_50 == 0):
                        CL_50 = (L_50 - lift_tare_50) * LBF_TO_NEWTONS / (dynpr_50 * PSF_TO_PA)
                    if not (np.isnan(D_50) or np.isnan(drag_tare_50) or np.isnan(dynpr_50) or dynpr_50 == 0):
                        CD_50 = (D_50 - drag_tare_50) * LBF_TO_NEWTONS / (dynpr_50 * PSF_TO_PA)
                    
                    # Calculate deltas
                    delta_CL = CL_60 - CL_50
                    delta_CD = CD_60 - CD_50
                    
                    print(f"[WT] Speed correction: {ref_60_name}@60mph CL={CL_60:.6f} CD={CD_60:.6f}, {ref_50_name}@50mph CL={CL_50:.6f} CD={CD_50:.6f}")
                    print(f"[WT] Speed correction deltas: delta_CL={delta_CL:.6f}, delta_CD={delta_CD:.6f}")
                    print(f"[WT] Will apply correction to unstable setpoint: {unstable_50_name}")

        # Find minimum CD from rows with valid w_cx (post speed-correction)
        min_cd = float('inf')
        for i, setpoint_row in enumerate(setpoints_data):
            try:
                # Get setpoint name
                sp_name = None
                sp_col_idx = _find_col_index('setpoint', map_columns)
                if sp_col_idx != -1:
                    sp_name = setpoint_row[sp_col_idx]
                
                # Exclude reference setpoints from min_CD calculation if speed_correction is configured
                if speed_correction and instability and sp_name:
                    ref_60_name = speed_correction.get('reference_60')
                    ref_50_name = speed_correction.get('reference_50')
                    if sp_name in [ref_60_name, ref_50_name]:
                        continue  # Skip reference setpoints
                
                # Get D, DYNPR, and w_cx values for this setpoint
                D = 0.0
                dynpr = 0.0
                wcx = 0.0
                
                # Extract D and DYNPR values
                if sp_name is not None and sp_name in per_setpoint_values:
                    D = per_setpoint_values[sp_name].get('D', 0.0)
                    dynpr = per_setpoint_values[sp_name].get('DYNPR', 0.0)
                elif 'D' in d1_column_indices and 'DYNPR' in d1_column_indices and i < len(d1_data):
                    row_index = min(i, len(d1_data) - 1)
                    D = to_float(d1_data[row_index][d1_column_indices['D']])
                    dynpr = to_float(d1_data[row_index][d1_column_indices['DYNPR']])
                
                # Extract w_cx, road_speed, and wind_speed from map setpoint
                wcx_idx = _find_col_index('w_cx', map_columns)
                road_speed_idx = _find_col_index('road_speed', map_columns)
                wind_speed_idx = _find_col_index('wind_speed', map_columns)
                if wcx_idx != -1:
                    wcx = to_float(setpoint_row[wcx_idx])
                road_speed = to_float(setpoint_row[road_speed_idx]) if road_speed_idx != -1 else 60.0
                wind_speed = to_float(setpoint_row[wind_speed_idx]) if wind_speed_idx != -1 else 60.0
                
                # Skip rows with wind_speed=0 (Zero/tare setpoints)
                if wind_speed == 0:
                    continue
                
                # Get appropriate drag tare for this speed
                drag_tare = get_tare_for_speed(road_speed, 'D')
                
                # Calculate CD if all values are valid
                if not (np.isnan(D) or np.isnan(drag_tare) or np.isnan(dynpr) or np.isnan(wcx) or dynpr == 0):
                    cd = (D - drag_tare) * LBF_TO_NEWTONS / (dynpr * PSF_TO_PA)
                    
                    # Apply speed correction if this is the unstable setpoint
                    if speed_correction and instability and unstable_50_name and sp_name == unstable_50_name:
                        if not np.isnan(cd):
                            cd = cd + delta_CD
                    
                    if not np.isnan(cd):
                        min_cd = min(min_cd, cd)
            except Exception:
                continue
        
        # Handle case where no valid CD values found
        if min_cd == float('inf'):
            min_cd = 0.0
        
        print(f"[WT] Minimum CD found for d1_processed (valid w_cx, wind_speed>0, post-correction, excl. references): {min_cd:.6f}")

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
                
                # Extract w_cz, w_cx, and road_speed from map setpoint
                wcz_idx = _find_col_index('w_cz', map_columns)
                wcx_idx = _find_col_index('w_cx', map_columns)
                road_speed_idx = _find_col_index('road_speed', map_columns)
                if wcz_idx != -1:
                    wcz = to_float(setpoint_row[wcz_idx])
                if wcx_idx != -1:
                    wcx = to_float(setpoint_row[wcx_idx])
                road_speed = to_float(setpoint_row[road_speed_idx]) if road_speed_idx != -1 else 60.0
                
                # Get appropriate tares for this setpoint's speed
                lift_tare = get_tare_for_speed(road_speed, 'L')
                drag_tare = get_tare_for_speed(road_speed, 'D')
                
                # Calculate CL, CD, CLW and CDW using intermediate steps
                # Calculate CL (Coefficient of Lift) first
                cl = float('nan')
                cd = float('nan')
                if not np.isnan(L) and not np.isnan(lift_tare) and not np.isnan(dynpr) and dynpr != 0:
                    cl = (L - lift_tare) * LBF_TO_NEWTONS / (dynpr * PSF_TO_PA)
                
                # Calculate CD (Coefficient of Drag) first
                if not np.isnan(D) and not np.isnan(drag_tare) and not np.isnan(dynpr) and dynpr != 0:
                    cd = (D - drag_tare) * LBF_TO_NEWTONS / (dynpr * PSF_TO_PA)
                
                # Apply speed correction if this is the unstable setpoint
                if speed_correction and instability and sp_name == speed_correction.get('unstable_50'):
                    cl_before = cl
                    cd_before = cd
                    if not np.isnan(cl):
                        cl = cl + delta_CL
                    if not np.isnan(cd):
                        cd = cd + delta_CD
                    print(f"[WT] Applied speed correction to {sp_name}: CL {cl_before:.6f} -> {cl:.6f}, CD {cd_before:.6f} -> {cd:.6f}")
                
                # Store CL and CD values and calculate weighted values
                if not np.isnan(cl):
                    cl_val = f"{cl:.6f}"
                    if not np.isnan(wcz):
                        clw = cl * wcz  # CLW = CL * w_cz
                        clw_val = f"{clw:.6f}"
                
                if not np.isnan(cd):
                    cd_val = f"{cd:.6f}"
                    if not np.isnan(wcx):
                        cdw = cd * wcx  # CDW = CD * w_cx (back to original formula)
                        cdw_val = f"{cdw:.6f}"
                
                # Per-row debug for short map while building d1_processed
                if map_name == "short":
                    try:
                        print(f"[WT][short][d1_processed] i={i} sp={sp_name} L={L} D={D} DYNPR={dynpr} w_cz={wcz} w_cx={wcx} CL={cl_val} CD={cd_val} CLW={clw_val} CDW={cdw_val}")
                    except Exception:
                        pass
                        
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
        subfolders = [f for f in os.listdir(folder) if os.path.isdir(os.path.join(folder, f)) and "Run0" in f]
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
                # Load d2 only if it exists
                d2_exists = os.path.exists(d2_path)
                if d2_exists:
                    d2, d2_colnames = load_d1_with_colnames(d2_path)
                else:
                    d2, d2_colnames = None, None
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

                    # Only create d2 dataset if d2 file exists
                    if d2_exists and d2 is not None:
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
                        # Validate row counts before creation and get min_Cx_weighting
                        sp_data_for_count, _sp_cols, min_Cx_weighting, instability, speed_correction = load_setpoints_from_map(selected_map)
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
                                ok, min_cd_msg = compute_weighted_coeffs_from_d1_processed(run_grp, map_name=selected_map, min_Cx_weighting=min_Cx_weighting, instability=instability, speed_correction=speed_correction)
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

                if d2_exists:
                    message = f"Imported {folder_name} successfully."
                else:
                    message = f"Imported {folder_name} successfully (d2.asc not found, skipped)."
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
                            def get_attr(key):
                                v = run_attrs.get(key, "")
                                if isinstance(v, bytes):
                                    v = v.decode()
                                return v
                            
                            def get_weighted_attr(key):
                                v = run_attrs.get(key, 0.0)
                                if isinstance(v, bytes):
                                    v = v.decode()
                                try:
                                    val = float(v)
                                    formatted = f"{val:.4f}"  # Changed to 4 decimal places
                                    return formatted
                                except (ValueError, TypeError):
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
                        def get_attr(key):
                            v = run_attrs.get(key, "")
                            if isinstance(v, bytes):
                                v = v.decode()
                            return v
                        
                        def get_weighted_attr(key):
                            v = run_attrs.get(key, 0.0)
                            if isinstance(v, bytes):
                                v = v.decode()
                            try:
                                val = float(v)
                                formatted = f"{val:.4f}"  # Changed to 4 decimal places
                                return formatted
                            except (ValueError, TypeError):
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
                            sp_data_for_count, _sp_cols, min_Cx_weighting_new, instability_new, speed_correction_new = load_setpoints_from_map(new_map)
                            map_rows = len(sp_data_for_count) if sp_data_for_count else 0
                            d1_rows = int(d1_data.shape[0]) if hasattr(d1_data, 'shape') and len(d1_data.shape) > 0 else (len(d1_data) if isinstance(d1_data, (list, np.ndarray)) else 0)
                            if map_rows > 0 and d1_rows > 0 and map_rows != d1_rows:
                                # Restore d1_processed with old map before returning error
                                if old_map:
                                    _, _, min_Cx_weighting_old, instability_old, speed_correction_old = load_setpoints_from_map(old_map)
                                    combined_data, combined_columns = create_d1_processed_data(old_map, d1_data, d1_columns, d1_structured, d1_structured_cols)
                                    if combined_data and combined_columns:
                                        combined_array = np.array(combined_data, dtype='S50')
                                        d1_processed_ds = run_grp.create_dataset("d1_processed", data=combined_array)
                                        d1_processed_ds.attrs["description"] = f"Setpoints from map: {old_map} with d1 data columns"
                                        d1_processed_ds.attrs["columns"] = np.array(combined_columns, dtype='S50')
                                        compute_weighted_coeffs_from_d1_processed(run_grp, map_name=old_map, min_Cx_weighting=min_Cx_weighting_old, instability=instability_old, speed_correction=speed_correction_old)
                                return f"Error: map has {map_rows} rows but d1 has {d1_rows} rows. Map remains: {old_map}"

                            # Create combined data with map setpoints and d1 columns
                            combined_data, combined_columns = create_d1_processed_data(new_map, d1_data, d1_columns, d1_structured, d1_structured_cols)
                            if combined_data and combined_columns:
                                combined_array = np.array(combined_data, dtype='S50')
                                d1_processed_ds = run_grp.create_dataset("d1_processed", data=combined_array)
                                d1_processed_ds.attrs["description"] = f"Setpoints from map: {new_map} with d1 data columns"
                                d1_processed_ds.attrs["columns"] = np.array(combined_columns, dtype='S50')
                                ok, min_cd_msg = compute_weighted_coeffs_from_d1_processed(run_grp, map_name=new_map, min_Cx_weighting=min_Cx_weighting_new, instability=instability_new, speed_correction=speed_correction_new)
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

 
