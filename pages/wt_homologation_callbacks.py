import os

import dash
from dash import html, dcc, Output, Input, State, callback_context
from dash.dependencies import ALL, Input, Output, State
import h5py
import numpy as np
import json
from dash import no_update
from dash import dash_table

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
                    {"name": "Delete", "id": "delete", "presentation": "markdown", "editable": False}
                ],
                data=[],
                dropdown={
                    "run_type": {
                        "options": [{"label": opt, "value": opt} for opt in run_type_options],
                        "clearable": True
                    }
                },
                editable=True,
                style_table={"overflowX": "auto", "overflowY": "visible"},
                style_cell={"textAlign": "left", "minWidth": "100px", "maxWidth": "250px", "whiteSpace": "normal"},
                style_cell_conditional=[
                    {"if": {"column_id": "delete"}, "width": "10px", "minWidth": "10px", "maxWidth": "10px", "textAlign": "center", "padding": "0", "overflow": "hidden"}
                ],
                css=[
                    {"selector": ".dash-cell.column-delete", "rule": "cursor: pointer; background: #ffeaea;"},
                    {"selector": ".dash-cell > div", "rule": "overflow: visible !important;"},
                    {"selector": ".Select-menu-outer", "rule": "display: block !important; z-index: 10000;"}
                ],
                style_header={"fontWeight": "bold", "backgroundColor": "#f5f5f5"},
                row_selectable="single",
                selected_rows=[],
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
                    colnames = lines[3].strip().split()
                    import io
                    data_lines = [line for i, line in enumerate(lines) if i > 4]
                    data = np.genfromtxt(io.StringIO(''.join(data_lines)), delimiter=None)
                    return data, colnames
                d1, d1_colnames = load_d1_with_colnames(d1_path)
                d2, d2_colnames = load_d1_with_colnames(d2_path)
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

                message = f"Imported {folder_name} successfully."
            except Exception as e:
                message = f"Error: {e}"
        # Always reload the run list after import
        h5_path = homologation["h5_path"] if homologation and homologation.get("h5_path") else None
        run_fields = []
        if h5_path:
            exists = os.path.exists(h5_path)
            if not exists:
                print("update_imported_runs_list: HDF5 file does not exist!")
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
                            table_data.append({
                                "run": run,
                                "description": get_attr("description"),
                                "weighted_Cz": get_attr("weighted_Cz"),
                                "weighted_Cx": get_attr("weighted_Cx"),
                                "offset_Cz": get_attr("offset_Cz"),
                                "offset_Cx": get_attr("offset_Cx"),
                                "run_type": rt_value,
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
                                    {"name": "Delete", "id": "delete", "presentation": "markdown", "editable": False}
                                ],
                                data=table_data,
                                dropdown={
                                    "run_type": {
                                        "options": [{"label": opt, "value": opt} for opt in run_type_options],
                                        "clearable": True
                                    }
                                },
                                editable=True,
                                style_table={"overflowX": "auto", "overflowY": "visible"},
                                style_cell={"textAlign": "left", "minWidth": "100px", "maxWidth": "250px", "whiteSpace": "normal"},
                                style_cell_conditional=[
                                    {"if": {"column_id": "delete"}, "width": "10px", "minWidth": "10px", "maxWidth": "10px", "textAlign": "center", "padding": "0", "overflow": "hidden"}
                                ],
                                css=[
                                    {"selector": ".dash-cell.column-delete", "rule": "cursor: pointer; background: #ffeaea;"},
                                    {"selector": ".dash-cell > div", "rule": "overflow: visible !important;"},
                                    {"selector": ".Select-menu-outer", "rule": "display: block !important; z-index: 10000;"}
                                ],
                                style_header={"fontWeight": "bold", "backgroundColor": "#f5f5f5"},
                                row_selectable="single",
                                selected_rows=[],
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
            print("update_imported_runs_list: HDF5 file does not exist!")
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

                        table_data.append({
                            "run": to_plain_type(run),
                            "description": to_plain_type(get_attr("description")),
                            "weighted_Cz": to_plain_type(get_attr("weighted_Cz")),
                            "weighted_Cx": to_plain_type(get_attr("weighted_Cx")),
                            "offset_Cz": to_plain_type(get_attr("offset_Cz")),
                            "offset_Cx": to_plain_type(get_attr("offset_Cx")),
                            "run_type": rt_value,
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
                                {"name": "Delete", "id": "delete", "presentation": "markdown", "editable": False}
                            ],
                            data=table_data,
                            dropdown={
                                "run_type": {
                                    "options": [{"label": opt, "value": opt} for opt in run_type_options],
                                    "clearable": True
                                }
                            },
                            editable=True,
                            style_table={"overflowX": "auto", "overflowY": "visible"},
                            style_cell={"textAlign": "left", "minWidth": "100px", "maxWidth": "250px", "whiteSpace": "normal"},
                            style_cell_conditional=[
                                {"if": {"column_id": "delete"}, "width": "10px", "minWidth": "10px", "maxWidth": "10px", "textAlign": "center", "padding": "0", "overflow": "hidden"}
                            ],
                            css=[
                                {"selector": ".dash-cell.column-delete", "rule": "cursor: pointer; background: #ffeaea;"},
                                {"selector": ".dash-cell > div", "rule": "overflow: visible !important;"},
                                {"selector": ".Select-menu-outer", "rule": "display: block !important; z-index: 10000;"}
                            ],
                            style_header={"fontWeight": "bold", "backgroundColor": "#f5f5f5"},
                            row_selectable="single",
                            selected_rows=[],
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
        
        return "Changes saved successfully"
    except Exception as e:
        return f"Error saving changes: {e}"

 
