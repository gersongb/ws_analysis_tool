import os

import dash
from dash import html, dcc, Output, Input, State, callback_context
from dash.dependencies import ALL, Input, Output, State
import h5py
import numpy as np
import json
from dash import no_update
from dash import dash_table

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
        
        # Combine map columns with d1 columns
        combined_columns = map_columns + d1_columns_to_append
        
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
    State("imported-runs-table", "data"),
    State("import-message-store", "data"),
    prevent_initial_call=False
)
def update_imported_runs_list(homologation, n_clicks_list, active_cell, table_data, last_message):
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
                    d2_ds = run_grp.create_dataset("d2", data=d2)
                    d2_ds.attrs["columns"] = np.array(d2_colnames, dtype='S')
                    run_grp.attrs["description"] = "no description available"
                    run_grp.attrs["weighted_Cz"] = 0.0
                    run_grp.attrs["weighted_Cx"] = 0.0
                    run_grp.attrs["offset_Cz"] = 0.0
                    run_grp.attrs["offset_Cx"] = 0.0
                    # Set run_type to the first option from run_plot_config
                    run_grp.attrs["run_type"] = run_type_options[0] if run_type_options else ""
                    # Set map to the first option from wt_maps
                    selected_map = map_options[0] if map_options else ""
                    run_grp.attrs["map"] = selected_map
                    
                    # Create d1_processed dataset with setpoints from the selected map and d1 data
                    if selected_map:
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
                            def get_attr(key):
                                v = run_attrs.get(key, "")
                                if isinstance(v, bytes):
                                    v = v.decode()
                                return v
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
                                "weighted_Cz": get_attr("weighted_Cz"),
                                "weighted_Cx": get_attr("weighted_Cx"),
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
                            "weighted_Cz": to_plain_type(get_attr("weighted_Cz")),
                            "weighted_Cx": to_plain_type(get_attr("weighted_Cx")),
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
                    
                    run_grp.attrs["map"] = new_map
                    
                    # Update d1_processed dataset if map has changed
                    if new_map != old_map:
                        # Remove existing d1_processed dataset if it exists
                        if "d1_processed" in run_grp:
                            del run_grp["d1_processed"]
                        
                        # Create new d1_processed dataset with setpoints from the new map
                        if new_map:
                            # Get d1 data and columns from the existing dataset
                            d1_data = []
                            d1_columns = []
                            if "d1" in run_grp:
                                d1_dataset = run_grp["d1"]
                                d1_data = d1_dataset[:]
                                if "columns" in d1_dataset.attrs:
                                    d1_columns = [col.decode() if isinstance(col, bytes) else col 
                                                for col in d1_dataset.attrs["columns"]]
                            
                            # Create combined data with map setpoints and d1 columns
                            combined_data, combined_columns = create_d1_processed_data(new_map, d1_data, d1_columns)
                            if combined_data and combined_columns:
                                # Convert structured data to numpy array
                                combined_array = np.array(combined_data, dtype='S50')
                                d1_processed_ds = run_grp.create_dataset("d1_processed", data=combined_array)
                                d1_processed_ds.attrs["description"] = f"Setpoints from map: {new_map} with d1 data columns"
                                # Store column names as attributes
                                d1_processed_ds.attrs["columns"] = np.array(combined_columns, dtype='S50')
                            else:
                                # Create empty dataset if no setpoints found
                                empty_setpoints = np.array([], dtype='S50').reshape(0, 0)
                                d1_processed_ds = run_grp.create_dataset("d1_processed", data=empty_setpoints)
                                d1_processed_ds.attrs["description"] = f"No setpoints found for map: {new_map}"
                                d1_processed_ds.attrs["columns"] = np.array([], dtype='S50')
                        else:
                            # Create empty dataset if no map selected
                            empty_setpoints = np.array([], dtype='S50').reshape(0, 0)
                            d1_processed_ds = run_grp.create_dataset("d1_processed", data=empty_setpoints)
                            d1_processed_ds.attrs["description"] = "No map selected"
                            d1_processed_ds.attrs["columns"] = np.array([], dtype='S50')
        
        return "Changes saved successfully"
    except Exception as e:
        return f"Error saving changes: {e}"

 
