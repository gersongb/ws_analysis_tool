import dash
from dash import html, dcc, callback, Input, Output, State
from dash.dependencies import MATCH
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
    dcc.Interval(id="live-wt-copy-interval", interval=20000, n_intervals=0, disabled=True),
    # Interval to refresh plots (always on) every 10s
    dcc.Interval(id="live-wt-plot-refresh-interval", interval=10000, n_intervals=0, disabled=False, max_intervals=-1),

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
        # Heartbeat to confirm interval ticks (debug)
        html.Div(id="live-wt-heartbeat", style={"marginBottom": "10px", "fontSize": "0.85rem", "color": "#888"}),

        # Performance Window (at top)
        dbc.Container([
            html.Div("Performance Window", style={"fontWeight": "bold", "marginBottom": "10px", "textAlign": "center", "fontSize": "1.2rem"}),
            html.Div(
                dcc.Graph(
                    id="live-wt-homologation-plot",
                    style={"width": "80%", "height": "100%", "minHeight": "0"},
                    config={"responsive": True}
                ),
                style={
                    "display": "flex",
                    "justifyContent": "center",
                    "alignItems": "center",
                    "height": "600px"
                }
            ),
            html.Div(id="live-wt-homologation-feedback", style={"marginTop": "10px", "color": "#cc0000", "textAlign": "center"})
        ], fluid=True, style={"padding": "20px", "border": "1px solid #ddd", "borderRadius": "5px", "marginBottom": "16px"}),

        # Tabs container
        dbc.Container([
            dcc.Tabs(
                id="live-wt-tabs-bottom",
                persistence=True,
                persistence_type="session"
            )
        ], fluid=True, style={"padding": "20px", "border": "1px solid #ddd", "borderRadius": "5px", "marginBottom": "16px"})
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
    print(f"[Live WT] Copy callback triggered: n={n}")
    if not homologation or not toggle_values or ("on" not in toggle_values):
        print(f"[Live WT] Copy skipped: toggle off or no homologation")
        return dash.no_update
    base_folder = homologation.get("base_folder")
    data_src = homologation.get("data_source_folder")
    if not base_folder or not data_src or not live_folder:
        print(f"[Live WT] Copy skipped: missing paths")
        return dash.no_update

    live_data_path = os.path.join(base_folder, "Live Data")
    live_d1_src = os.path.join(data_src, live_folder, "d1.asc")
    live_target = os.path.join(live_data_path, "Live_D1.asc")

    try:
        if os.path.isfile(live_d1_src) and os.path.isdir(live_data_path):
            shutil.copyfile(live_d1_src, live_target)
            ts = datetime.now().strftime("%H:%M:%S")
            print(f"[Live WT] ✓ File copied at {ts}: {live_d1_src} → {live_target}")
            return f"Live Comparison running. Last update: {ts}"
    except Exception as e:
        print(f"[Live WT] ✗ Copy failed: {e}")
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
            # Small per-tick margin toggle to force a minimal layout change
            try:
                r_margin = 20 if (isinstance(refresh_token, int) and (refresh_token % 2 == 0)) else 21
            except Exception:
                r_margin = 20
            fig.update_layout(
                title=ch,
                xaxis_title="Point Number",
                margin=dict(l=40, r=r_margin, t=40, b=30),
                height=250,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            # Force client re-render without affecting visuals (no dummy trace to keep indices stable for extendData)
            if refresh_token is not None:
                invisible_anno = dict(
                    text=str(refresh_token),
                    xref="paper", yref="paper", x=1.05, y=-0.15,
                    showarrow=False,
                    font=dict(size=1, color="rgba(0,0,0,0)"),
                    opacity=0
                )
                fig.update_layout(
                    datarevision=str(refresh_token),
                    meta={"tick": str(refresh_token)},
                    annotations=[invisible_anno]
                )
            # Use pattern-matching ID to support extendData updates per graph
            graph_id = {"type": "live-graph", "group": group_name, "channel": ch}
            graphs.append(dcc.Graph(id=graph_id, figure=fig, config={"responsive": True}, style={"marginBottom": "10px"}))
        tabs.append(dcc.Tab(label=group_name, children=graphs))
    return tabs


# ExtendData callback for bottom graphs: append the latest live point for each channel
@callback(
    Output({"type": "live-graph", "group": MATCH, "channel": MATCH}, "extendData"),
    Input("live-wt-plot-refresh-interval", "n_intervals"),
    State("current-homologation-store", "data"),
    State("live-wt-live-dropdown", "value"),
    State({"type": "live-graph", "group": MATCH, "channel": MATCH}, "id"),
    State({"type": "live-graph", "group": MATCH, "channel": MATCH}, "figure"),
    prevent_initial_call=True
)
def _extend_live_graphs(_tick, homologation, live_folder, graph_id, fig_state):
    try:
        if not homologation or not live_folder:
            return dash.no_update
        data_src = homologation.get("data_source_folder")
        if not data_src:
            return dash.no_update
        live_path = os.path.join(data_src, live_folder, "d1.asc")
        series = _load_d1_series(live_path)
        # Obtain the channel key from the graph ID (more reliable than title)
        ch_name = None
        try:
            ch_name = graph_id.get("channel") if isinstance(graph_id, dict) else None
        except Exception:
            ch_name = None
        if not ch_name or ch_name not in series:
            return dash.no_update
        y_arr = series.get(ch_name)
        if y_arr is None or len(y_arr) == 0:
            return dash.no_update
        # X axis uses Point Number if available, else incremental index
        x_arr = series.get("Point Number")

        # Determine trace index for Live trace in this figure
        trace_idx = -1
        data_traces = fig_state.get("data", []) if isinstance(fig_state, dict) else []
        for i, tr in enumerate(data_traces):
            if isinstance(tr, dict) and tr.get("name") == "Live":
                trace_idx = i
                break
        # If Live trace is not present in this figure yet, skip updating to avoid corrupting Reference
        if trace_idx == -1:
            return dash.no_update

        # Current plotted length for the Live trace
        current_len = 0
        try:
            current_x = data_traces[trace_idx].get("x", [])
            # Support list or numpy array
            if hasattr(current_x, "__len__"):
                current_len = len(current_x)
            else:
                current_len = 0
        except Exception:
            current_len = 0

        total_len = len(y_arr)
        if current_len >= total_len:
            return dash.no_update

        # Append all new points since last length; cap batch to avoid huge updates
        max_batch = 200
        end_idx = min(total_len, current_len + max_batch)
        new_y_seq = [float(v) for v in y_arr[current_len:end_idx]]
        if isinstance(x_arr, np.ndarray) and x_arr.shape[0] == len(y_arr):
            new_x_seq = [float(v) for v in x_arr[current_len:end_idx]]
        else:
            # Use sample index when no proper X is present
            new_x_seq = list(range(current_len, end_idx))

        extend = {"x": [new_x_seq], "y": [new_y_seq]}
        max_points = 1000
        return (extend, [trace_idx], max_points)
    except Exception:
        return dash.no_update


@callback(
    Output("live-wt-homologation-plot", "figure"),
    Output("live-wt-homologation-feedback", "children"),
    Input("current-homologation-store", "data"),
    Input("live-wt-plot-refresh-interval", "n_intervals"),
    Input("live-wt-comparison-toggle", "value"),
    Input("live-wt-reference-dropdown", "value"),
    Input("live-wt-live-dropdown", "value"),
    prevent_initial_call=False
)
def update_live_homologation_plot(homologation_data, _plot_tick, toggle_values, ref_folder, live_folder):
    """Update the homologation window plot with reference and live data points."""
    from datetime import datetime
    print(f"\n[Live WT] === Plot callback triggered at {datetime.now().strftime('%H:%M:%S')} | tick={_plot_tick} ===")
    
    if not homologation_data or "reference_folder" not in homologation_data:
        return go.Figure(), "No homologation loaded."
    
    wt_json_path = os.path.join(homologation_data["reference_folder"], "config", "wt.json")
    if not os.path.exists(wt_json_path):
        return go.Figure(), f"wt.json not found at {wt_json_path}"
    
    try:
        with open(wt_json_path, "r") as f:
            wt_data = json.load(f)
        windshear = wt_data.get("windshear")
        if not windshear:
            return go.Figure(), "wt.json missing 'windshear' key."
        cx = windshear.get("Cx")
        cz = windshear.get("Cz")
        if cx is None or cz is None:
            return go.Figure(), "Missing 'Cx' or 'Cz' in 'windshear'."
        if not isinstance(cx, list) or not isinstance(cz, list):
            return go.Figure(), "'Cx' and 'Cz' must be lists."
        if len(cx) != len(cz):
            return go.Figure(), "'Cx' and 'Cz' lengths do not match."
        if len(cx) == 0:
            return go.Figure(), "'Cx' and 'Cz' are empty."
        
        # Create figure with performance window
        fig = go.Figure(data=go.Scatter(x=cx, y=cz, mode='lines+markers', name='Performance Window', 
                                       line=dict(color='black'), showlegend=True))
        
        # Track min/max for axis ranges
        x_min, x_max = min(cx), max(cx)
        y_min, y_max = min(cz), max(cz)
        
        # If live comparison is active, add reference and live data points
        if toggle_values and ("on" in toggle_values) and homologation_data.get("h5_path"):
            h5_path = homologation_data["h5_path"]
            
            # Load weighted values from HDF5 imported runs table
            try:
                import h5py
                with h5py.File(h5_path, "r") as h5f:
                    if "wt_runs" in h5f:
                        # Initialize variables
                        ref_wcx, ref_wcz, ref_ocx, ref_ocz = None, None, None, None
                        live_wcx, live_wcz, live_ocx, live_ocz = None, None, None, None
                        final_ref_cx, final_ref_cz = None, None
                        
                        # Helper function to get attributes from HDF5
                        def get_run_weighted_values(run_name):
                            if run_name not in h5f["wt_runs"]:
                                return None, None, None, None
                            run_attrs = h5f["wt_runs"][run_name].attrs
                            try:
                                weighted_cx = float(run_attrs.get("weighted_Cx", 0.0))
                                weighted_cz = float(run_attrs.get("weighted_Cz", 0.0))
                                offset_cx = float(run_attrs.get("offset_Cx", 0.0))
                                offset_cz = float(run_attrs.get("offset_Cz", 0.0))
                                return weighted_cx, weighted_cz, offset_cx, offset_cz
                            except (ValueError, TypeError):
                                return None, None, None, None
                        
                        # Get reference run values
                        if ref_folder:
                            ref_wcx, ref_wcz, ref_ocx, ref_ocz = get_run_weighted_values(ref_folder)
                            if ref_wcx is not None and ref_wcz is not None:
                                # Apply offsets
                                final_ref_cx = ref_wcx + ref_ocx
                                final_ref_cz = ref_wcz + ref_ocz
                                
                                # Add reference point
                                fig.add_trace(go.Scatter(
                                    x=[final_ref_cx],
                                    y=[final_ref_cz],  # Keep original sign, y-axis will be inverted
                                    mode='markers',
                                    name='Reference',
                                    marker=dict(color='blue', size=12, symbol='circle'),
                                    text=[f"Ref: {ref_folder}"],
                                    hovertemplate='<b>%{text}</b><br>Cx: %{x:.4f}<br>Cz: %{y:.4f}<extra></extra>'
                                ))
                                
                                # Update axis ranges
                                x_min = min(x_min, final_ref_cx)
                                x_max = max(x_max, final_ref_cx)
                                y_min = min(y_min, final_ref_cz)
                                y_max = max(y_max, final_ref_cz)
                        
                        # Get live run values
                        if live_folder:
                            live_wcx, live_wcz, live_ocx, live_ocz = get_run_weighted_values(live_folder)
                            if live_wcx is not None and live_wcz is not None:
                                # Apply offsets
                                final_live_cx = live_wcx + live_ocx
                                final_live_cz = live_wcz + live_ocz
                                
                                # Add live point
                                fig.add_trace(go.Scatter(
                                    x=[final_live_cx],
                                    y=[final_live_cz],  # Keep original sign, y-axis will be inverted
                                    mode='markers',
                                    name='Live',
                                    marker=dict(color='red', size=12, symbol='diamond'),
                                    text=[f"Live: {live_folder}"],
                                    hovertemplate='<b>%{text}</b><br>Cx: %{x:.4f}<br>Cz: %{y:.4f}<extra></extra>'
                                ))
                                
                                # Update axis ranges
                                x_min = min(x_min, final_live_cx)
                                x_max = max(x_max, final_live_cx)
                                y_min = min(y_min, final_live_cz)
                                y_max = max(y_max, final_live_cz)
                        
                        # Compute predicted live performance based on L and D corrections
                        print(f"[Live WT] Checking prediction conditions: ref={ref_folder}, live={live_folder}, ref_wcx={ref_wcx}, ref_wcz={ref_wcz}, final_ref_cx={final_ref_cx}, final_ref_cz={final_ref_cz}")
                        if ref_folder and live_folder and ref_wcx is not None and ref_wcz is not None and final_ref_cx is not None and final_ref_cz is not None:
                            print(f"[Live WT] Attempting to compute prediction...")
                            try:
                                # Load reference d1_processed data to get L, D, and weights
                                if ref_folder in h5f["wt_runs"] and "d1_processed" in h5f["wt_runs"][ref_folder]:
                                    ref_ds = h5f["wt_runs"][ref_folder]["d1_processed"]
                                    ref_cols = [col.decode() if isinstance(col, bytes) else col 
                                               for col in ref_ds.attrs.get("columns", [])]
                                    ref_data = ref_ds[:]
                                    
                                    # Find column indices
                                    def find_col_idx(cols, name):
                                        try:
                                            return cols.index(name)
                                        except ValueError:
                                            return None
                                    
                                    L_idx = find_col_idx(ref_cols, "L")
                                    D_idx = find_col_idx(ref_cols, "D")
                                    wcx_idx = find_col_idx(ref_cols, "w_cx")
                                    wcz_idx = find_col_idx(ref_cols, "w_cz")
                                    frh_idx = find_col_idx(ref_cols, "frh")
                                    rrh_idx = find_col_idx(ref_cols, "rrh")
                                    
                                    print(f"[Live WT] Column indices: L={L_idx}, D={D_idx}, wcx={wcx_idx}, wcz={wcz_idx}, frh={frh_idx}, rrh={rrh_idx}")
                                    if all(idx is not None for idx in [L_idx, D_idx, wcx_idx, wcz_idx]):
                                        print(f"[Live WT] All columns found, extracting arrays from {len(ref_data)} rows")
                                        # Extract reference arrays (don't filter, just convert)
                                        ref_L = np.array([float(row[L_idx]) if row[L_idx] else 0.0 for row in ref_data])
                                        ref_D = np.array([float(row[D_idx]) if row[D_idx] else 0.0 for row in ref_data])
                                        ref_wcx_weights = np.array([float(row[wcx_idx]) if row[wcx_idx] else 0.0 for row in ref_data])
                                        ref_wcz_weights = np.array([float(row[wcz_idx]) if row[wcz_idx] else 0.0 for row in ref_data])
                                        print(f"[Live WT] Extracted arrays: L={len(ref_L)}, D={len(ref_D)}, wcx={len(ref_wcx_weights)}, wcz={len(ref_wcz_weights)}")
                                        
                                        # Load live L and D from locally copied file
                                        live_L = None
                                        live_D = None
                                        
                                        base_folder = homologation_data.get("base_folder")
                                        print(f"[Live WT] Base folder: {base_folder}")
                                        if base_folder:
                                            live_data_dir = os.path.join(base_folder, "Live Data")
                                            live_d1_path = os.path.join(live_data_dir, "Live_D1.asc")
                                            print(f"[Live WT] Looking for live d1 at: {live_d1_path}")
                                            if os.path.exists(live_d1_path):
                                                # Check file modification time
                                                import time
                                                mod_time = os.path.getmtime(live_d1_path)
                                                print(f"[Live WT] Live d1 file exists | Last modified: {time.ctime(mod_time)} | Size: {os.path.getsize(live_d1_path)} bytes")
                                                
                                                # Force fresh read by opening file directly
                                                live_series = _load_d1_series(live_d1_path)
                                                print(f"[Live WT] Live series columns: {list(live_series.keys())}")
                                                if "L" in live_series and "D" in live_series:
                                                    live_L = live_series["L"]
                                                    live_D = live_series["D"]
                                                    print(f"[Live WT] Loaded live L and D: {len(live_L)} rows | L[0]={live_L[0] if len(live_L) > 0 else 'N/A'}")
                                                else:
                                                    print(f"[Live WT] L or D not found in live series")
                                            else:
                                                print(f"[Live WT] Live d1 file does not exist (toggle may be off or not copied yet)")
                                        else:
                                            print(f"[Live WT] Missing base_folder")
                                        
                                        # If live data is available, use comparable rows only
                                        print(f"[Live WT] Checking dimensions: live_L={len(live_L) if live_L is not None else 'None'}, ref_L={len(ref_L)}, live_D={len(live_D) if live_D is not None else 'None'}, ref_D={len(ref_D)}")
                                        if live_L is not None and live_D is not None and len(live_L) > 0 and len(live_D) > 0:
                                            # Handle partial data: only use rows available in both datasets
                                            n_comparable = min(len(live_L), len(ref_L), len(live_D), len(ref_D), len(ref_wcx_weights), len(ref_wcz_weights))
                                            print(f"[Live WT] Using {n_comparable} comparable rows (partial data handling)")
                                            
                                            if n_comparable > 0:
                                                # Slice arrays to comparable length
                                                ref_L_comp = ref_L[:n_comparable]
                                                ref_D_comp = ref_D[:n_comparable]
                                                live_L_comp = live_L[:n_comparable]
                                                live_D_comp = live_D[:n_comparable]
                                                ref_wcx_weights_comp = ref_wcx_weights[:n_comparable]
                                                ref_wcz_weights_comp = ref_wcz_weights[:n_comparable]
                                                
                                                # Find the row with FRH=30 and RRH=30 to apply additional 20% weight
                                                frh30_rrh30_row = None
                                                if frh_idx is not None and rrh_idx is not None:
                                                    try:
                                                        # Search for FRH=30 and RRH=30 in comparable rows
                                                        for i in range(n_comparable):
                                                            frh_val = float(ref_data[i][frh_idx]) if ref_data[i][frh_idx] else 0.0
                                                            rrh_val = float(ref_data[i][rrh_idx]) if ref_data[i][rrh_idx] else 0.0
                                                            
                                                            if abs(frh_val - 30.0) < 0.1 and abs(rrh_val - 30.0) < 0.1:
                                                                frh30_rrh30_row = i
                                                                print(f"[Live WT] Found FRH=30, RRH=30 at row {i}")
                                                                break
                                                    except Exception as e:
                                                        print(f"[Live WT] Error finding FRH=30, RRH=30 row: {e}")
                                                
                                                # Adjust Cx weights: add 20% extra weight to FRH=30, RRH=30 row
                                                adjusted_wcx_weights = ref_wcx_weights_comp.copy()
                                                if frh30_rrh30_row is not None:
                                                    adjusted_wcx_weights[frh30_rrh30_row] += 20.0
                                                    print(f"[Live WT] Added 20% weight to row {frh30_rrh30_row}: original={ref_wcx_weights_comp[frh30_rrh30_row]:.2f}, adjusted={adjusted_wcx_weights[frh30_rrh30_row]:.2f}")
                                                
                                                # Normalize weights to sum to 100
                                                wcz_sum = np.sum(ref_wcz_weights_comp)
                                                wcx_sum = np.sum(adjusted_wcx_weights)
                                                
                                                if wcz_sum > 0:
                                                    normalized_wcz = (ref_wcz_weights_comp / wcz_sum) * 100
                                                else:
                                                    normalized_wcz = ref_wcz_weights_comp
                                                
                                                if wcx_sum > 0:
                                                    normalized_wcx = (adjusted_wcx_weights / wcx_sum) * 100
                                                else:
                                                    normalized_wcx = adjusted_wcx_weights
                                                
                                                print(f"[Live WT] Normalized weights: Cz sum={np.sum(normalized_wcz):.2f}%, Cx sum={np.sum(normalized_wcx):.2f}%")
                                                
                                                # Compute correction factors for each comparable row
                                                # Avoid division by zero
                                                L_corrections = np.where(ref_L_comp != 0, (live_L_comp - ref_L_comp) / ref_L_comp, 0)
                                                D_corrections = np.where(ref_D_comp != 0, (live_D_comp - ref_D_comp) / ref_D_comp, 0)
                                                
                                                # Apply weighted corrections to predict live coefficients
                                                # Use normalized weights (sum to 100%)
                                                weighted_L_correction = np.sum(L_corrections * normalized_wcz) / 100.0
                                                weighted_D_correction = np.sum(D_corrections * normalized_wcx) / 100.0
                                                
                                                # Compute predicted values
                                                predicted_cz = final_ref_cz * (1 + weighted_L_correction)
                                                predicted_cx = final_ref_cx * (1 + weighted_D_correction)
                                                
                                                # Add predicted live point
                                                fig.add_trace(go.Scatter(
                                                    x=[predicted_cx],
                                                    y=[predicted_cz],  # Keep original sign, y-axis will be inverted
                                                    mode='markers',
                                                    name='Predicted Live',
                                                    marker=dict(color='orange', size=12, symbol='star'),
                                                    text=[f"Predicted: {live_folder}"],
                                                    hovertemplate='<b>%{text}</b><br>Cx: %{x:.4f}<br>Cz: %{y:.4f}<extra></extra>'
                                                ))
                                                
                                                # Update axis ranges
                                                x_min = min(x_min, predicted_cx)
                                                x_max = max(x_max, predicted_cx)
                                                y_min = min(y_min, predicted_cz)
                                                y_max = max(y_max, predicted_cz)
                                                
                                                print(f"[Live WT] ✓ Predicted live: Cx={predicted_cx:.4f}, Cz={predicted_cz:.4f}")
                                                print(f"[Live WT] ✓ L correction: {weighted_L_correction:.4f}, D correction: {weighted_D_correction:.4f}")
                                            else:
                                                print(f"[Live WT] ✗ No comparable rows available")
                                        else:
                                            print(f"[Live WT] ✗ Live data missing or empty")
                                    else:
                                        print(f"[Live WT] ✗ Required columns not found in d1_processed")
                                else:
                                    print(f"[Live WT] ✗ Reference run not found in HDF5 or missing d1_processed")
                            except Exception as e:
                                print(f"[Live WT] Error computing predicted live performance: {e}")
                                import traceback
                                traceback.print_exc()
            except Exception as e:
                print(f"[Live WT] Error loading weighted values from HDF5: {e}")
        
        # Expand x/y limits by ±0.03
        # Small per-tick margin toggle to force a minimal layout change
        try:
            r_margin_top = 120 if (int(_plot_tick) % 2 == 0) else 119
        except Exception:
            r_margin_top = 120
        fig.update_layout(
            xaxis_title="Cx",
            yaxis_title="Cz",
            xaxis_range=[x_min - 0.03, x_max + 0.03],
            yaxis_range=[y_max + 0.03, y_min - 0.03],  # Reversed order to invert y-axis
            legend=dict(x=1.02, y=1, xanchor='left', yanchor='top', orientation='v'),
            margin=dict(r=r_margin_top),
            autosize=True,
            height=None,
            width=None,
            # Tie revisions to refresh tick to force client update on each interval
            datarevision=str(_plot_tick),
            meta={"tick": str(_plot_tick)},
            # Add invisible per-tick annotation to guarantee a real layout change
            annotations=[dict(
                text=str(_plot_tick),
                xref="paper", yref="paper", x=1.05, y=-0.15,
                showarrow=False,
                font=dict(size=1, color="rgba(0,0,0,0)"),
                opacity=0
            )]
        )
        # Add an invisible per-tick dummy trace so data array changes
        fig.add_trace(go.Scatter(x=[0], y=[0], mode="markers", name=f"_tick_{_plot_tick}",
                                 marker=dict(opacity=0), line=dict(color="rgba(0,0,0,0)"),
                                 showlegend=False, hoverinfo="skip"))
        return fig, ""
    except Exception as e:
        return go.Figure(), f"Error reading wt.json: {e}"


@callback(
    Output("live-wt-tabs-bottom", "children"),
    Input("live-wt-comparison-toggle", "value"),
    State("current-homologation-store", "data"),
    State("live-wt-reference-dropdown", "value"),
    State("live-wt-live-dropdown", "value"),
    prevent_initial_call=False
)
def refresh_tabs(toggle_values, homologation, ref_folder, live_folder):
    import time
    print(f"[Live WT] refresh_tabs called at {time.strftime('%H:%M:%S')} | toggle={toggle_values}")
    # If toggle is off, clear all plots
    if not toggle_values or ("on" not in toggle_values):
        print(f"[Live WT] Toggle is OFF, clearing plots")
        return []
    if not homologation or not homologation.get("base_folder"):
        print(f"[Live WT] No homologation, clearing plots")
        return []
    base_folder = homologation["base_folder"]
    data_src = homologation.get("data_source_folder")
    live_data_dir = os.path.join(base_folder, "Live Data")
    ref_path = os.path.join(live_data_dir, "Reference_D1.asc")
    # Always re-read Live directly from source folder
    live_path = os.path.join(data_src, live_folder, "d1.asc") if (data_src and live_folder) else None
    print(f"[Live WT] Building tabs | ref={ref_path} | live={live_path}")
    tabs = _build_tabs_content(ref_path, live_path)
    print(f"[Live WT] Built {len(tabs)} tabs")
    return tabs


# Heartbeat: show ticks to confirm intervals are firing and reaching client
@callback(
    Output("live-wt-heartbeat", "children"),
    Input("live-wt-plot-refresh-interval", "n_intervals"),
    prevent_initial_call=False
)
def _live_wt_heartbeat(n):
    try:
        from datetime import datetime
        ts = datetime.now().strftime("%H:%M:%S")
        return f"Heartbeat tick: {n} at {ts}"
    except Exception:
        return f"Heartbeat tick: {n}"
