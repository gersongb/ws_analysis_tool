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

layout = dbc.Container([
    html.H1("Setup"),
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
    dcc.Store(id="current-homologation-store", storage_type="session"),
], fluid=True)

# Combined callback for create/load homologation
@dash.callback(
    Output("create-homologation-feedback", "children"),
    Output("load-homologation-feedback", "children"),
    Output("load-homologation-dropdown", "options"),
    Output("current-homologation-store", "data"),
    Input("create-homologation-btn", "n_clicks"),
    Input("load-homologation-btn", "n_clicks"),
    State("manufacturer-dropdown", "value"),
    State("homologation-date-picker", "date"),
    State("new-base-folder", "value"),
    State("name-identifier", "value"),
    State("load-homologation-dropdown", "value"),
    prevent_initial_call=True
)
def handle_homologation(create_n, load_n, manufacturer, homologation_date, new_base, name_identifier, load_value):
    from datetime import datetime
    from data_processing.hdf5_functions import create_homologation_hdf5
    ctx = dash.callback_context
    create_msg = ""
    load_msg = ""
    data = dash.no_update

    if not ctx.triggered:
        return "", "", get_homologation_options(), dash.no_update

    button_id = ctx.triggered[0]["prop_id"].split(".")[0]

    if button_id == "create-homologation-btn":
        if not manufacturer or not new_base or not homologation_date:
            create_msg = "Please select a manufacturer, date, and enter a base folder."
        else:
            # Parse date and build session name
            try:
                date_obj = datetime.strptime(homologation_date[:10], "%Y-%m-%d")
                session_name = f"{manufacturer}_{date_obj.strftime('%m-%Y')}"
                if name_identifier:
                    session_name = f"{session_name}_{name_identifier}"
            except Exception:
                create_msg = "Invalid date format."
                return create_msg, load_msg, get_homologation_options(), data
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
                # Create config subfolder and copy config files
                config_src_dir = os.path.join(os.path.dirname(__file__), "..", "config")
                config_dst_dir = os.path.join(ref_folder, "config")
                os.makedirs(config_dst_dir, exist_ok=True)
                for fname in ["wt.json", "brake_blanking.json", "run_plot_config.json"]:
                    shutil.copy2(os.path.join(config_src_dir, fname), os.path.join(config_dst_dir, fname))
            except Exception as e:
                create_msg = f"Failed to create folders: {e}"
            else:
                data = {
                    "manufacturer": manufacturer,
                    "homologation_date": date_obj.strftime("%Y-%m-%d"),
                    "session_name": session_name,
                    "name_identifier": name_identifier,
                    "base_folder": new_base,
                    "reference_folder": ref_folder,
                    "wt_maps": wt_maps_path,
                    "data": data_path,
                    "raw": raw_path
                }
                filename = f"{session_name}.json"
                path = os.path.join(HOMOLOGATIONS_DIR, filename)
                h5_path = os.path.join(data_path, f"{session_name}.h5")
                try:
                    with open(path, "w") as f:
                        json.dump(data, f, indent=2)
                    # Create HDF5 file with setup as attributes and wt_runs group
                    create_homologation_hdf5(h5_path, data)
                    create_msg = f"Created homologation '{session_name}'"
                except Exception as e:
                    create_msg = f"Failed to save homologation: {e}"
    elif button_id == "load-homologation-btn":
        if not load_value:
            load_msg = "Please select a homologation to load."
        else:
            loaded = load_homologation_data(load_value)
            if loaded:
                load_msg = f"Loaded homologation '{loaded.get('session_name', load_value)}'"
                data = loaded
            else:
                load_msg = "Failed to load homologation."

    return create_msg, load_msg, get_homologation_options(), data

# Callback to display homologation info
@dash.callback(
    Output("homologation-info-panel", "children"),
    Input("current-homologation-store", "data"),
)
def display_homologation_info(data):
    if not data:
        return "No homologation loaded."
    # Display key reference info
    items = [html.Div([html.B(k+":"), " ", str(v)]) for k, v in data.items()]
    return items
