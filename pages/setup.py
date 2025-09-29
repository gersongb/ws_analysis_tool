import dash
from dash import html, dcc, Input, Output, State
import dash_bootstrap_components as dbc
import os
import json

dash.register_page(__name__, path="/setup", name="Setup")

HOMOLOGATIONS_DIR = os.path.join(os.path.dirname(__file__), "..", ".homologations")
if not os.path.exists(HOMOLOGATIONS_DIR):
    os.makedirs(HOMOLOGATIONS_DIR)

MANUFACTURERS_FILE = os.path.join(os.path.dirname(__file__), "..", "config", "manufacturers.json")
MAPS_FILE = os.path.join(os.path.dirname(__file__), "..", "config", "maps.json")

# Ensure manufacturers file exists
if not os.path.exists(MANUFACTURERS_FILE):
    import json
    import os
    os.makedirs(os.path.dirname(MANUFACTURERS_FILE), exist_ok=True)
    with open(MANUFACTURERS_FILE, "w") as f:
        json.dump(INITIAL_MANUFACTURERS_DICT, f, indent=2)

def get_manufacturers():
    try:
        with open(MANUFACTURERS_FILE, "r") as f:
            data = json.load(f)
            return list(data.keys())
    except Exception:
        return list(INITIAL_MANUFACTURERS_DICT.keys())

def get_homologation_options():
    files = [f for f in os.listdir(HOMOLOGATIONS_DIR) if f.endswith(".json")]
    return [{"label": f.replace(".json", ""), "value": f} for f in files]

def load_homologation_data(filename):
    path = os.path.join(HOMOLOGATIONS_DIR, filename)
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return None

# Utility to get wind tunnel keys from wt.json
import_path = os.path.join(os.path.dirname(__file__), "..", "config", "wt.json")
def get_wind_tunnel_keys():
    try:
        with open(import_path, "r") as f:
            data = json.load(f)
            return list(data.keys())
    except Exception:
        return []

layout = dbc.Container([
    html.H2("Setup"),
    html.Hr(),
    dbc.Row([
        dbc.Col([
            html.H4("Load Existing Homologation"),
            dcc.Dropdown(id="load-homologation-dropdown", options=get_homologation_options(), placeholder="Select a homologation to load"),
            dbc.Button("Load", id="load-homologation-btn", color="primary", n_clicks=0, style={"marginTop": "10px"}),
            html.Div(id="load-homologation-feedback", style={"marginTop": "10px", "color": "#008800"}),
        ], width=6),
        dbc.Col([
            html.H4("Create New Homologation"),
            html.Label("Base folder:"),
            html.Br(),
            dcc.Input(id="new-base-folder", type="text", placeholder="Enter or paste folder path", style={"width": "90%", "marginBottom": "10px"}),
            html.Br(),
            html.Label("Data Source Folder:"),
            html.Br(),
            dcc.Input(id="data-source-folder", type="text", placeholder="Enter or paste data source folder path", style={"width": "90%", "marginBottom": "10px"}),
            html.Br(),
            html.Label("Wind Tunnel:"),
            html.Br(),
            dcc.Dropdown(
                id="wind-tunnel-dropdown",
                options=[{"label": k, "value": k} for k in get_wind_tunnel_keys()],
                placeholder="Select wind tunnel",
                style={"width": "90%", "marginBottom": "10px"}
            ),
            html.Br(),
            html.Label("Manufacturer:"),
            html.Br(),
            dcc.Dropdown(
                id="manufacturer-dropdown",
                options=[{"label": m, "value": m} for m in get_manufacturers()],
                placeholder="Select manufacturer",
                style={"width": "90%", "marginBottom": "10px"}
            ),
            html.Br(),
            html.Label("Homologation Date:"),
            html.Br(),
            dcc.DatePickerSingle(
                id="homologation-date-picker",
                display_format="DD/MM/YYYY",
                placeholder="Select a date",
                style={"marginBottom": "10px"}
            ),
            html.Br(),
            html.Label("Name Identifier:"),
            html.Br(),
            dcc.Input(id="name-identifier", type="text", placeholder="Enter identifier (optional)", style={"width": "90%", "marginBottom": "10px"}),
            html.Br(),
            dbc.Button("Create", id="create-homologation-btn", color="success", n_clicks=0, style={"marginTop": "10px"}),
            html.Div(id="create-homologation-feedback", style={"marginTop": "10px", "color": "#008800"}),
        ], width=6)
    ]),
    html.Hr(),
    html.H4("Homologation Reference Information"),
    html.Div(id="homologation-info-panel", style={"marginTop": "10px", "padding": "10px", "border": "1px solid #ccc", "borderRadius": "5px"}),
    dcc.Store(id="current-homologation-store", storage_type="local"),

    html.Hr(),
    html.H4("Modify Data Source Folder for Current Homologation"),
    dbc.Row([
        dbc.Col([
            dcc.Input(id="modify-data-source-folder-input", type="text", placeholder="Enter new data source folder", style={"width": "70%", "marginRight": "10px"}),
            dbc.Button("Update", id="modify-data-source-folder-btn", color="primary", n_clicks=0),
            html.Div(id="modify-data-source-folder-feedback", style={"marginTop": "10px", "color": "#008800"})
        ], width=12)
    ])
], fluid=True)

# Always keep the info panel in sync with the store, even after reload/navigation
@dash.callback(
    Output("homologation-info-panel", "children"),
    Input("current-homologation-store", "data"),
)
def keep_info_panel_in_sync(data):
    if not data:
        return "No homologation loaded."
    return [html.Div([html.B(f"{k}:"), " ", str(v)]) for k, v in data.items()]

# Combined callback for create/load homologation
@dash.callback(
    Output("create-homologation-feedback", "children"),
    Output("load-homologation-feedback", "children"),
    Output("load-homologation-dropdown", "options"),
    Output("modify-data-source-folder-feedback", "children"),
    Output("current-homologation-store", "data"),
    Input("create-homologation-btn", "n_clicks"),
    Input("load-homologation-btn", "n_clicks"),
    Input("modify-data-source-folder-btn", "n_clicks"),
    State("manufacturer-dropdown", "value"),
    State("homologation-date-picker", "date"),
    State("new-base-folder", "value"),
    State("data-source-folder", "value"),
    State("wind-tunnel-dropdown", "value"),
    State("name-identifier", "value"),
    State("load-homologation-dropdown", "value"),
    State("modify-data-source-folder-input", "value"),
    State("current-homologation-store", "data"),
    prevent_initial_call=True
)
def unified_homologation_callback(
    create_n, load_n, modify_n,
    manufacturer, homologation_date, new_base, data_source_folder, wind_tunnel, name_identifier, load_value,
    modify_folder_value, current_data
):
    import dash
    import json, os
    from datetime import datetime
    ctx = dash.callback_context
    create_msg = ""
    load_msg = ""
    modify_msg = dash.no_update
    info_panel = dash.no_update
    dropdown_options = get_homologation_options()
    data = current_data

    if not ctx.triggered:
        return "", "", dropdown_options, dash.no_update, dash.no_update

    button_id = ctx.triggered[0]["prop_id"].split(".")[0]

    # Create Homologation
    if button_id == "create-homologation-btn":
        if not manufacturer or not new_base or not homologation_date or not data_source_folder or not wind_tunnel or not name_identifier:
            create_msg = "Please fill all fields."
        else:
            try:
                date_obj = datetime.strptime(homologation_date[:10], "%Y-%m-%d")
                session_name = f"{manufacturer}_{date_obj.strftime('%m-%Y')}_{wind_tunnel}_{name_identifier}"
            except Exception:
                create_msg = "Invalid date format."
                return create_msg, load_msg, dropdown_options, modify_msg, data
            ref_folder_name = session_name.replace(" ", "_")
            ref_folder = os.path.join(new_base, ref_folder_name)
            import shutil
            try:
                os.makedirs(ref_folder, exist_ok=True)
                wt_maps_path = os.path.join(ref_folder, "wt_maps")
                data_path = os.path.join(ref_folder, "data")
                os.makedirs(wt_maps_path, exist_ok=True)
                os.makedirs(data_path, exist_ok=True)
                raw_path = os.path.join(data_path, "raw")
                os.makedirs(raw_path, exist_ok=True)
                config_src_dir = os.path.join(os.path.dirname(__file__), "..", "config")
                config_dst_dir = os.path.join(ref_folder, "config")
                os.makedirs(config_dst_dir, exist_ok=True)
                for fname in ["wt.json", "brake_blanking.json", "run_plot_config.json"]:
                    shutil.copy2(os.path.join(config_src_dir, fname), os.path.join(config_dst_dir, fname))
                # Generate one JSON file per map into wt_maps, derived from config/maps.json
                try:
                    if os.path.exists(MAPS_FILE):
                        with open(MAPS_FILE, "r") as mf:
                            maps_payload = json.load(mf)
                        # Write each top-level map into its own JSON file
                        for map_name, map_content in maps_payload.items():
                            out_path = os.path.join(wt_maps_path, f"{map_name}.json")
                            with open(out_path, "w") as out_f:
                                json.dump(map_content, out_f, indent=2)
                    else:
                        # If maps.json is missing, create an empty placeholder to avoid breaking downstream flows
                        placeholder_path = os.path.join(wt_maps_path, "README.txt")
                        with open(placeholder_path, "w") as pf:
                            pf.write("No maps.json found at creation time. Place per-map JSON files here.")
                except Exception as map_e:
                    # Non-fatal: proceed with homologation creation but report in feedback
                    create_msg = f"Created homologation '{session_name}', but failed to create wt_maps files: {map_e}"
                h5_path = os.path.join(data_path, f"{session_name}.h5")
                data = {
                    "manufacturer": manufacturer,
                    "homologation_date": date_obj.strftime("%Y-%m-%d"),
                    "session_name": session_name,
                    "name_identifier": name_identifier,
                    "base_folder": new_base,
                    "reference_folder": ref_folder,
                    "wt_maps": wt_maps_path,
                    "data": data_path,
                    "raw": raw_path,
                    "data_source_folder": data_source_folder,
                    "wind_tunnel": wind_tunnel,
                    "h5_path": h5_path
                }
                filename = f"{session_name}.json"
                path = os.path.join(os.path.dirname(__file__), "..", ".homologations", filename)
                with open(path, "w") as f:
                    json.dump(data, f, indent=2)
                # Create HDF5 file with setup as attributes and wt_runs group
                h5_path = os.path.join(data_path, f"{session_name}.h5")
                try:
                    from data_processing.hdf5_functions import create_homologation_hdf5
                    create_homologation_hdf5(h5_path, data)
                except Exception as e:
                    create_msg = f"Homologation JSON saved, but failed to create HDF5: {e}"
                # Only set a success message if not already set by a maps/HDF5 warning above
                if not create_msg:
                    create_msg = f"Created homologation '{session_name}'"
                dropdown_options = get_homologation_options()
            except Exception as e:
                create_msg = f"Failed to save homologation: {e}"

    # Load Homologation
    elif button_id == "load-homologation-btn":
        if not load_value:
            load_msg = "Please select a homologation to load."
        else:
            loaded = load_homologation_data(load_value)
            if loaded:
                # Add h5_path if not present
                session_name = loaded.get("session_name")
                data_path = loaded.get("data")
                if session_name and data_path:
                    h5_path = os.path.join(data_path, f"{session_name}.h5")
                    loaded["h5_path"] = h5_path
                load_msg = f"Loaded homologation '{loaded.get('session_name', load_value)}'"
                data = loaded
            else:
                load_msg = "Failed to load homologation."

    # Modify Data Source Folder
    elif button_id == "modify-data-source-folder-btn":
        if not modify_n or not modify_folder_value or not current_data:
            modify_msg = dash.no_update
        else:
            data = dict(current_data)
            data["data_source_folder"] = modify_folder_value
            session_name = data.get("session_name")
            if not session_name:
                modify_msg = "No homologation loaded."
            else:
                try:
                    homologation_dir = os.path.join(os.path.dirname(__file__), "..", ".homologations")
                    file_path = os.path.join(homologation_dir, f"{session_name}.json")
                    if not os.path.exists(file_path):
                        for f in os.listdir(homologation_dir):
                            if f.endswith(".json") and session_name in f:
                                file_path = os.path.join(homologation_dir, f)
                                break
                    with open(file_path, "w") as f:
                        json.dump(data, f, indent=2)
                    # Also update the HDF5 file with the new data
                    try:
                        data_path = data.get("data", None)
                        session_name = data.get("session_name", None)
                        if data_path and session_name:
                            h5_path = os.path.join(data_path, f"{session_name}.h5")
                            from data_processing.hdf5_functions import create_homologation_hdf5
                            create_homologation_hdf5(h5_path, data)
                        modify_msg = "Data source folder and HDF5 updated!"
                    except Exception as e:
                        modify_msg = f"Data source folder updated, but failed to update HDF5: {e}"
                except Exception as e:
                    modify_msg = f"Failed to update: {e}"

    # Always update info panel
    # info_panel is not an output; return only 5 values as required
    return create_msg, load_msg, dropdown_options, modify_msg, data
