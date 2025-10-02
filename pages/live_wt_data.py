import dash
from dash import html, dcc, callback, Input, Output, State
import dash_bootstrap_components as dbc
import os
import shutil
from datetime import datetime
import json
import numpy as np
import plotly.graph_objs as go

dash.register_page(__name__, path="/live-wt-data", name="Live WT Data")

layout = dbc.Container([
    html.H2("Live WT Data"),

    # Store for current homologation data (shared across pages)
    dcc.Store(id="current-homologation-store", storage_type="local"),

    # Interval to refresh dropdowns periodically
    dcc.Interval(id="live-wt-refresh-interval", interval=5000, n_intervals=0),
    # Interval to run live copying (disabled until toggle on)
    dcc.Interval(id="live-wt-copy-interval", interval=30000, n_intervals=0, disabled=True),
    # Interval to refresh plots (always on) every 10s
    dcc.Interval(id="live-wt-plot-refresh-interval", interval=10000, n_intervals=0),

    # Main inner container
    dbc.Container([
        dbc.Row([
            dbc.Col([
                html.Label("Reference", style={"fontWeight": "bold", "marginBottom": "5px"}),
                dcc.Dropdown(
                    id="live-wt-reference-dropdown",
                    placeholder="Select reference run...",
                    style={"width": "300px"},
                    persistence=True,
                    persistence_type="local"
                )
            ], width="auto"),
            dbc.Col([
                html.Label("Live", style={"fontWeight": "bold", "marginBottom": "5px"}),
                dcc.Dropdown(
                    id="live-wt-live-dropdown",
                    placeholder="Select live run...",
                    style={"width": "300px"},
                    persistence=True,
                    persistence_type="local"
                )
            ], width="auto"),
            dbc.Col([
                html.Label("\u00A0", style={"display": "block", "marginBottom": "5px"}),  # spacer to align switch
                dbc.Checklist(
                    id="live-wt-comparison-toggle",
                    options=[{"label": "Live Comparison", "value": "on"}],
                    value=[],
                    switch=True,
                    inline=True,
                    style={"whiteSpace": "nowrap"},
                    persistence=True,
                    persistence_type="local"
                )
            ], width="auto"),
        ], style={"marginBottom": "20px"}),

        # Status/debug message
        html.Div(id="live-wt-status", style={"marginBottom": "10px", "fontSize": "0.9rem", "color": "#666"}),

        # Tabs container (top)
        dbc.Container([
            dcc.Tabs(id="live-wt-tabs-top")
        ], fluid=True, style={"padding": "20px", "border": "1px solid #ddd", "borderRadius": "5px", "marginBottom": "16px"}),

        # Second container with duplicated tabs (as requested)
        dbc.Container([
            dcc.Tabs(id="live-wt-tabs-bottom")
        ], fluid=True, style={"padding": "20px", "border": "1px solid #ddd", "borderRadius": "5px"})
    ], fluid=True, style={"padding": "20px", "border": "1px solid #ddd", "borderRadius": "5px"})
], fluid=True)

# Callback to populate dropdowns with Run0 folders from data source
@callback(
    Output("live-wt-reference-dropdown", "options"),
    Output("live-wt-live-dropdown", "options"),
    Output("live-wt-status", "children"),
    Input("current-homologation-store", "data"),
    Input("live-wt-refresh-interval", "n_intervals"),
    prevent_initial_call=False
)
def update_live_wt_dropdowns(homologation, n_intervals):
    if not homologation or not homologation.get("data_source_folder"):
        return [], [], "No homologation loaded. Please go to Setup tab and load a homologation."
    
    folder = homologation["data_source_folder"]
    if not os.path.isdir(folder):
        return [], [], f"Data source folder not found: {folder}"
    
    try:
        # Get all subfolders containing "Run0" in the name
        subfolders = [f for f in os.listdir(folder) if os.path.isdir(os.path.join(folder, f)) and "Run0" in f]
        # Sort by modification time (newest first)
        subfolders = sorted(
            subfolders,
            key=lambda f: os.path.getmtime(os.path.join(folder, f)),
            reverse=True
        )
        # Create dropdown options
        options = [{"label": f, "value": f} for f in subfolders]
        status_msg = f"Found {len(subfolders)} run folders in data source."
        return options, options, status_msg
    except Exception as e:
        print(f"[Live WT] Error loading folders: {e}")
        return [], [], f"Error loading folders: {e}"

# --- Live Comparison logic ---

@callback(
    Output("live-wt-copy-interval", "disabled"),
    Output("live-wt-status", "children", allow_duplicate=True),
    Output("live-wt-status", "style", allow_duplicate=True),
    State("current-homologation-store", "data"),
    Input("live-wt-comparison-toggle", "value"),
    State("live-wt-reference-dropdown", "value"),
    State("live-wt-live-dropdown", "value"),
    prevent_initial_call=True
)
def handle_live_comparison(homologation, toggle_values, ref_folder, live_folder):
    status_style = {"marginBottom": "10px", "fontSize": "0.9rem", "color": "#666"}
    if not homologation or not homologation.get("base_folder") or not homologation.get("data_source_folder"):
        return True, "Live Comparison unavailable: no homologation loaded.", {**status_style, "color": "#c00"}

    base_folder = homologation["base_folder"]
    data_src = homologation["data_source_folder"]
    live_data_path = os.path.join(base_folder, "Live Data")

    toggle_on = bool(toggle_values) and ("on" in toggle_values)

    # Turn OFF: stop interval and delete Live Data folder
    if not toggle_on:
        try:
            if os.path.isdir(live_data_path):
                shutil.rmtree(live_data_path)
        except Exception as e:
            # Keep running, just report
            return True, f"Stopped Live Comparison. Failed to delete Live Data: {e}", {**status_style, "color": "#c00"}
        return True, "Stopped Live Comparison. Live Data removed.", status_style

    # Turn ON: validate selections and setup
    if not ref_folder or not live_folder:
        return True, "Select both Reference and Live runs to start Live Comparison.", {**status_style, "color": "#c00"}

    ref_d1 = os.path.join(data_src, ref_folder, "d1.asc")
    live_d1 = os.path.join(data_src, live_folder, "d1.asc")
    if not os.path.isfile(ref_d1):
        return True, f"Reference D1.asc not found in '{ref_folder}'.", {**status_style, "color": "#c00"}
    if not os.path.isfile(live_d1):
        return True, f"Live D1.asc not found in '{live_folder}'.", {**status_style, "color": "#c00"}

    # Ensure Live Data folder exists
    os.makedirs(live_data_path, exist_ok=True)

    # Copy reference once if not already copied
    ref_target = os.path.join(live_data_path, "Reference_D1.asc")
    if not os.path.isfile(ref_target):
        try:
            shutil.copy2(ref_d1, ref_target)
        except Exception as e:
            return True, f"Failed to copy Reference: {e}", {**status_style, "color": "#c00"}

    # Immediately copy live once on toggle ON
    live_target = os.path.join(live_data_path, "Live_D1.asc")
    try:
        shutil.copyfile(live_d1, live_target)
    except Exception as e:
        return True, f"Failed to copy Live: {e}", {**status_style, "color": "#c00"}

    # Enable interval for live copying
    banner = html.Div([
        html.Div(f"Live Comparison running | Ref: {ref_folder} | Live: {live_folder}"),
        html.Div(f"Live Data: {live_data_path}", style={"fontSize": "0.85rem", "color": "#888"})
    ])
    return False, banner, status_style


@callback(
    Output("live-wt-status", "children", allow_duplicate=True),
    Input("live-wt-copy-interval", "n_intervals"),
    State("current-homologation-store", "data"),
    State("live-wt-comparison-toggle", "value"),
    State("live-wt-reference-dropdown", "value"),
    State("live-wt-live-dropdown", "value"),
    prevent_initial_call=True
)
def perform_periodic_live_copy(n, homologation, toggle_values, ref_folder, live_folder):
    if not homologation or not toggle_values or ("on" not in toggle_values):
        return dash.no_update
    base_folder = homologation.get("base_folder")
    data_src = homologation.get("data_source_folder")
    if not base_folder or not data_src or not live_folder:
        return dash.no_update

    live_data_path = os.path.join(base_folder, "Live Data")
    live_d1_src = os.path.join(data_src, live_folder, "d1.asc")
    live_target = os.path.join(live_data_path, "Live_D1.asc")

    try:
        if os.path.isfile(live_d1_src) and os.path.isdir(live_data_path):
            shutil.copyfile(live_d1_src, live_target)
            ts = datetime.now().strftime("%H:%M:%S")
            return f"Live Comparison running. Last update: {ts}"
    except Exception as e:
        return f"Live copy failed: {e}"
    return dash.no_update

# -------- Plot building helpers and callbacks --------

def _load_live_channels_config():
    try:
        cfg_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "config", "live_channels.json"))
        if not os.path.exists(cfg_path):
            return {}
        with open(cfg_path, "r") as f:
            return json.load(f)
    except Exception:
        return {}


def _load_d1_series(file_path):
    """Load D1.asc as dict of column name -> numpy array.
    Ignore lines 1,2,3,5 (1-indexed), use line 4 as headers, and data from line 6 onwards.
    """
    if not os.path.isfile(file_path):
        print(f"[Live WT] D1 file not found: {file_path}")
        return {}
    try:
        import time
        mod_time = os.path.getmtime(file_path)
        print(f"[Live WT] Loading D1: {file_path} | mod_time={time.ctime(mod_time)}")
        with open(file_path, "r", errors="ignore") as f:
            lines = f.readlines()
        if len(lines) < 5:
            return {}
        # headers on 4th line (1-indexed), i.e., index 3
        headers = [h.strip() for h in lines[3].strip().split('\t')]
        # data from line 6 onwards (index > 4). This effectively ignores 1,2,3,5.
        data_lines = [line for i, line in enumerate(lines) if i > 4]
        import io
        arr = np.genfromtxt(io.StringIO(''.join(data_lines)), delimiter='\t')
        if arr.ndim == 1:
            arr = arr.reshape(-1, len(headers)) if len(headers) > 1 else arr.reshape(-1, 1)
        series = {}
        for idx, name in enumerate(headers):
            try:
                series[name] = arr[:, idx]
            except Exception:
                pass
        print(f"[Live WT] Loaded {len(series)} channels, {arr.shape[0] if arr.ndim > 0 else 0} rows")
        return series
    except Exception as e:
        print(f"[Live WT] Error loading D1 {file_path}: {e}")
        return {}


def _build_tabs_content(ref_path, live_path, refresh_token=None):
    """Build list of dcc.Tab components based on live_channels.json and available data.
    Reads Reference and Live from the explicit file paths provided.
    """
    ref_series = _load_d1_series(ref_path) if ref_path else {}
    live_series = _load_d1_series(live_path) if live_path else {}
    channels_cfg = _load_live_channels_config()

    tabs = []
    for group_name, channel_list in channels_cfg.items():
        graphs = []
        for ch in channel_list:
            ref_y = ref_series.get(ch)
            live_y = live_series.get(ch)
            if ref_y is None and live_y is None:
                # Show placeholder if channel not found
                graphs.append(html.Div(f"Channel '{ch}' not found in D1 files.", style={"color": "#999", "marginBottom": "8px"}))
                continue
            fig = go.Figure()
            # X axis: Point Number if present, otherwise sample index
            x_ref = ref_series.get("Point Number") if ref_series else None
            x_live = live_series.get("Point Number") if live_series else None
            if ref_y is not None:
                xr = x_ref if isinstance(x_ref, np.ndarray) and x_ref.shape[0] == len(ref_y) else np.arange(len(ref_y))
                fig.add_trace(go.Scatter(x=xr, y=ref_y, mode="lines", name="Reference"))
            if live_y is not None:
                xl = x_live if isinstance(x_live, np.ndarray) and x_live.shape[0] == len(live_y) else np.arange(len(live_y))
                fig.add_trace(go.Scatter(x=xl, y=live_y, mode="lines", name="Live"))
            fig.update_layout(
                title=ch,
                xaxis_title="Point Number",
                margin=dict(l=40, r=20, t=40, b=30),
                height=250,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            # Force client re-render without affecting visuals
            if refresh_token is not None:
                fig.update_layout(datarevision=str(refresh_token))
            # Use unique id to force complete graph replacement on each refresh
            graph_id = f"live-graph-{group_name}-{ch}-{refresh_token}" if refresh_token is not None else f"live-graph-{group_name}-{ch}"
            graphs.append(dcc.Graph(id=graph_id, figure=fig, style={"marginBottom": "10px"}))
        tabs.append(dcc.Tab(label=group_name, children=graphs))
    return tabs


@callback(
    Output("live-wt-tabs-top", "children"),
    Output("live-wt-tabs-bottom", "children"),
    Input("live-wt-plot-refresh-interval", "n_intervals"),
    Input("live-wt-comparison-toggle", "value"),
    State("current-homologation-store", "data"),
    State("live-wt-reference-dropdown", "value"),
    State("live-wt-live-dropdown", "value"),
    prevent_initial_call=False
)
def refresh_tabs(_plot_tick, toggle_values, homologation, ref_folder, live_folder):
    import time
    print(f"[Live WT] refresh_tabs called at {time.strftime('%H:%M:%S')} | tick={_plot_tick} | toggle={toggle_values}")
    # If toggle is off, clear all plots
    if not toggle_values or ("on" not in toggle_values):
        print(f"[Live WT] Toggle is OFF, clearing plots")
        return [], []
    if not homologation or not homologation.get("base_folder"):
        print(f"[Live WT] No homologation, clearing plots")
        return [], []
    base_folder = homologation["base_folder"]
    data_src = homologation.get("data_source_folder")
    live_data_dir = os.path.join(base_folder, "Live Data")
    ref_path = os.path.join(live_data_dir, "Reference_D1.asc")
    # Always re-read Live directly from source folder
    live_path = os.path.join(data_src, live_folder, "d1.asc") if (data_src and live_folder) else None
    print(f"[Live WT] Building tabs | ref={ref_path} | live={live_path}")
    tabs = _build_tabs_content(ref_path, live_path, refresh_token=_plot_tick)
    print(f"[Live WT] Built {len(tabs)} tabs")
    return tabs, tabs
