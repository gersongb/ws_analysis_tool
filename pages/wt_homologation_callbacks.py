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
    State("import-message-store", "data"),
    prevent_initial_call=True
)
def update_imported_runs_list(homologation, n_clicks_list, last_message):
    from dash import callback_context, no_update
    import os
    ctx = callback_context
    

    message = None
    # --- Load run_plot_config.json and set run_type_options at the top ---
    run_plot_config_path = None
    if homologation and "reference_folder" in homologation:
        config_dst_dir = os.path.join(homologation["reference_folder"], "config")
        candidate = os.path.join(config_dst_dir, "run_plot_config.json")
        if os.path.exists(candidate):
            run_plot_config_path = candidate
    if not run_plot_config_path:
        run_plot_config_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "config", "run_plot_config.json")
        )
    with open(run_plot_config_path, "r") as f:
        run_plot_config = json.load(f)
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
                    d2_ds = run_grp.create_dataset("d2", data=d2, compression="gzip")
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
            print(f"update_imported_runs_list: h5_path={h5_path!r}, exists={exists}")
            if not exists:
                print("update_imported_runs_list: HDF5 file does not exist!")
            try:
                with h5py.File(h5_path, "r") as h5f:
                    print(f"update_imported_runs_list: h5f keys={list(h5f.keys())}")
                    if "wt_runs" in h5f:
                        runs = list(h5f["wt_runs"].keys())
                        print(f"update_imported_runs_list: found runs={runs}")
                        table_data = []
                        for run in runs:
                            run_attrs = h5f["wt_runs"][run].attrs
                            def get_attr(key):
                                v = run_attrs.get(key, "")
                                if isinstance(v, bytes):
                                    v = v.decode()
                                return v
                            rt_value = get_attr("run_type")
                            if not rt_value and run_type_options:
                                rt_value = run_type_options[0]
                                print(f"update_imported_runs_list: run_type not found for {run}, using {rt_value}")
                            table_data.append({
                                "run": run,
                                "description": get_attr("description"),
                                "weighted_Cz": get_attr("weighted_Cz"),
                                "weighted_Cx": get_attr("weighted_Cx"),
                                "offset_Cz": get_attr("offset_Cz"),
                                "offset_Cx": get_attr("offset_Cx"),
                                "run_type": rt_value,
                            })
                        run_fields = [
                            dash_table.DataTable(
                                id="imported-runs-table",
                                columns=[
                                    {"name": "Run", "id": "run"},
                                    {"name": "Description", "id": "description"},
                                    {"name": "Weighted Cz", "id": "weighted_Cz"},
                                    {"name": "Weighted Cx", "id": "weighted_Cx"},
                                    {"name": "Offset Cz", "id": "offset_Cz"},
                                    {"name": "Offset Cx", "id": "offset_Cx"},
                                    {"name": "Run Type", "id": "run_type"},
                                ],
                                data=table_data,
                                style_table={"overflowX": "auto"},
                                style_cell={"textAlign": "left", "minWidth": "100px", "maxWidth": "250px", "whiteSpace": "normal"},
                                style_header={"fontWeight": "bold", "backgroundColor": "#f5f5f5"},
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
        print(f"update_imported_runs_list: h5_path={h5_path!r}, exists={exists}")
        if not exists:
            print("update_imported_runs_list: HDF5 file does not exist!")
        try:
            with h5py.File(h5_path, "r") as h5f:
                print(f"update_imported_runs_list: h5f keys={list(h5f.keys())}")
                if "wt_runs" in h5f:
                    runs = list(h5f["wt_runs"].keys())
                    print(f"update_imported_runs_list: found runs={runs}")
                    table_data = []
                    for run in runs:
                        run_attrs = h5f["wt_runs"][run].attrs
                        def get_attr(key):
                            v = run_attrs.get(key, "")
                            if isinstance(v, bytes):
                                v = v.decode()
                            return v
                        table_data.append({
                            "run": run,
                            "description": get_attr("description"),
                            "weighted_Cz": get_attr("weighted_Cz"),
                            "weighted_Cx": get_attr("weighted_Cx"),
                            "offset_Cz": get_attr("offset_Cz"),
                            "offset_Cx": get_attr("offset_Cx"),
                            "run_type": get_attr("run_type"),
                        })


                    run_fields = [
                        dash_table.DataTable(
                            id="imported-runs-table",
                            columns=[
                                {"name": "Run", "id": "run"},
                                {"name": "Description", "id": "description"},
                                {"name": "Weighted Cz", "id": "weighted_Cz"},
                                {"name": "Weighted Cx", "id": "weighted_Cx"},
                                {"name": "Offset Cz", "id": "offset_Cz"},
                                {"name": "Offset Cx", "id": "offset_Cx"},
                                {"name": "Run Type", "id": "run_type", "presentation": "dropdown"},
                            ],
                            data=table_data,
                            dropdown={
                                "run_type": {
                                    "options": [{"label": opt, "value": opt} for opt in run_type_options]
                                }
                            },
                            editable=True,
                            style_table={"overflowX": "auto"},
                            style_cell={"textAlign": "left", "minWidth": "100px", "maxWidth": "250px", "whiteSpace": "normal"},
                            style_header={"fontWeight": "bold", "backgroundColor": "#f5f5f5"},
                            page_size=20,
                        )
                    ]


        except Exception as e:
            last_message = f"Error: {e}"
 
    return run_fields, last_message
