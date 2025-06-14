import os

import dash
from dash import html, dcc, Output, Input, State, callback_context
from dash.dependencies import ALL, Input, Output, State
import h5py
import numpy as np
from dash import no_update

# Callback to show run folders in data source folder with Import buttons
@dash.callback(
    Output("run-folder-list", "children"),
    Input("current-homologation-store", "data"),
    Input("run-folder-refresh-interval", "n_intervals"),
)
def update_run_folder_list(homologation, n_intervals):
    from dash import html
    import os
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
    ctx = callback_context
    

    message = None
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
                message = f"Imported {folder_name} successfully."
            except Exception as e:
                message = f"Error: {e}"
        # Always reload the run list after import
        h5_path = homologation["h5_path"] if homologation and homologation.get("h5_path") else None
        run_fields = []
        if h5_path:
            try:
                with h5py.File(h5_path, "r") as h5f:
                    if "wt_runs" in h5f:
                        runs = list(h5f["wt_runs"].keys())
                        for run in runs:
                            desc = ""
                            if "description" in h5f["wt_runs"][run].attrs:
                                desc = h5f["wt_runs"][run].attrs["description"].decode() if isinstance(h5f["wt_runs"][run].attrs["description"], bytes) else h5f["wt_runs"][run].attrs["description"]
                            run_fields.append(
                                html.Div([
                                    html.Span(run, style={"fontWeight": "bold", "marginRight": "10px"}),
                                    dcc.Input(id={"type": "run-desc-input", "index": run}, value=desc, placeholder="Enter description", debounce=True, style={"width": "300px", "marginRight": "10px"})
                                ], style={"marginBottom": "8px", "display": "flex", "alignItems": "center"})
                            )
            except Exception as e:
                message = f"Error: {e}"
        # Show success/error message, else show importing message
        return run_fields, (message if message else importing_msg)

    # Not triggered by import icon: just update run list, keep previous message
    h5_path = homologation["h5_path"] if homologation and homologation.get("h5_path") else None
    run_fields = []
    if h5_path:
        try:
            with h5py.File(h5_path, "r") as h5f:
                if "wt_runs" in h5f:
                    runs = list(h5f["wt_runs"].keys())
                    for run in runs:
                        desc = ""
                        if "description" in h5f["wt_runs"][run].attrs:
                            desc = h5f["wt_runs"][run].attrs["description"].decode() if isinstance(h5f["wt_runs"][run].attrs["description"], bytes) else h5f["wt_runs"][run].attrs["description"]
                        run_fields.append(
                            html.Div([
                                html.Span(run, style={"fontWeight": "bold", "marginRight": "10px"}),
                                dcc.Input(id={"type": "run-desc-input", "index": run}, value=desc, placeholder="Enter description", debounce=True, style={"width": "300px", "marginRight": "10px"})
                            ], style={"marginBottom": "8px", "display": "flex", "alignItems": "center"})
                        )
        except Exception as e:
            last_message = f"Error: {e}"
    return run_fields, last_message
