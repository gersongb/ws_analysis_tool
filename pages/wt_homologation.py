# ---- Imports ----
import os
# ---- Standard Library Imports ----
import os
import json

# ---- Third-Party Imports ----
import dash_bootstrap_components as dbc
import h5py
import numpy as np
import plotly.graph_objs as go

# ---- Dash Core Imports ----
import dash
from dash import html, dcc, Output, Input
from dash.dependencies import ALL
from dash import dash_table
# Ensure callbacks are registered for this page
import pages.wt_homologation_callbacks

# ---- Page Registration ----
dash.register_page(__name__, path="/wt-homologation", name="WT Homologation")

# Load run_plot_config.json for dropdown options
run_plot_config_path = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "config", "run_plot_config.json")
)
with open(run_plot_config_path, "r") as f:
    run_plot_config = json.load(f)
run_type_options = list(run_plot_config.keys())

# ---- Layout ----
layout = dbc.Container([
    # Page title
    html.H2("WT Homologation"),


    # Data stores
    dcc.Store(id="current-homologation-store", storage_type="local"),
    dcc.Store(id="wt-homologation-plot-store"),
    dcc.Store(id="import-message-store"),


    # Top row: two containers
    dbc.Row([
        # Left container: run folders
        dbc.Col([
            html.Div([
                html.Div("Run Folders in Data Source", style={"fontWeight": "bold", "marginBottom": "10px", "textAlign": "center", "fontSize": "1.2rem"}),
                html.Div(id="run-folder-list"),
                html.Div(id="import-feedback", style={"marginTop": "10px", "color": "#007700"}),
                dcc.Interval(id="run-folder-refresh-interval", interval=10000, n_intervals=0),
            ],
            style={
                "padding": "10px",
                "border": "2px solid #888",
                "borderRadius": "8px",
                "background": "#fafbfc",
                "height": "700px",
                "overflowY": "auto",
                "display": "flex",
                "flexDirection": "column",
                "justifyContent": "flex-start"
            }
            ),
        ], width=2),  
        
        # Right container: performance window plot
        dbc.Col([
            html.Div([
                html.Div("Performance Window", style={"fontWeight": "bold", "marginBottom": "10px", "textAlign": "center", "fontSize": "1.2rem"}),
                html.Div(
                    dcc.Graph(
                        id="wt-homologation-plot",
                        style={"width": "80%", "height": "100%", "minHeight": "0"},
                        config={"responsive": True}
                    ),
                    style={
                        "display": "flex",
                        "justifyContent": "center",
                        "alignItems": "center",
                        "height": "100%"
                    }
                ),
                html.Div(id="wt-homologation-plot-feedback", style={"marginTop": "10px", "color": "#cc0000"})
            ],
            style={
                "height": "700px",
                "border": "2px solid #888",
                "borderRadius": "8px",
                "background": "#fafbfc",
                "padding": "10px",
                "overflow": "hidden",
                "display": "flex",
                "flexDirection": "column"
            }
            ),
        ], width=10),  
    ], className="mb-3"),
    
    # Bottom wide container: imported runs and description fields
    dbc.Row([
        dbc.Col([
            html.Div([
                html.H5("Imported Runs"),
                html.Div(
                    id="imported-runs-list",
                    children=[
                        dash_table.DataTable(
                            id="imported-runs-table",
                            columns=[
                                {"name": "Run", "id": "run", "editable": False},
                                {"name": "Description", "id": "description", "editable": True},
                                {"name": "Weighted Cz", "id": "weighted_Cz", "editable": False},
                                {"name": "Weighted Cx", "id": "weighted_Cx", "editable": False},
                                {"name": "Offset Cz", "id": "offset_Cz", "editable": True},
                                {"name": "Offset Cx", "id": "offset_Cx", "editable": True},
                                {"name": "Run Type", "id": "run_type", "editable": True, "presentation": "dropdown"},
                                {"name": "Map", "id": "map", "editable": True, "presentation": "dropdown"},
                                {"name": "", "id": "delete", "presentation": "markdown", "editable": False},
                            ],
                            data=[],
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
                                {"selector": ".dash-cell.column-delete", "rule": "cursor: pointer; background: #ffeaea;"}
                            ],
                            style_header={"fontWeight": "bold", "backgroundColor": "#f5f5f5"},
                            style_header_conditional=[
                                {"if": {"column_id": c}, "textAlign": "center"}
                                for c in ["weighted_Cz", "weighted_Cx", "offset_Cz", "offset_Cx", "delete"]
                            ],
                            page_size=20,
                        )
                    ]
                ),
                
            ], style={
                "padding": "10px",
                "border": "2px solid #888",
                "borderRadius": "8px",
                "background": "#fafbfc",
                "overflow": "visible",
                "minHeight": "460px"
            })
        ], width=12),
    ], className="mb-3"),

    # Debug/messages area below imported runs
    dbc.Row([
        dbc.Col([
            html.Div(
                id="wt-homologation-message-area",
                style={
                    "marginTop": "10px",
                    "color": "#007700",
                    "fontSize": "0.97rem",
                    "background": "#f5f5f5",
                    "border": "1.5px solid #bbb",
                    "borderRadius": "8px",
                    "padding": "12px",
                    "minHeight": "36px"
                }
            )
        ], width=12),
    ], className="mb-3"),

], fluid=True)


# ---- Register Callbacks ----
from . import wt_homologation_callbacks

from dash import callback_context, Output, Input, State

from dash.dependencies import ALL, State
import h5py
import numpy as np

@dash.callback(
    Output("wt-homologation-plot", "figure"),
    Output("wt-homologation-plot-feedback", "children"),
    Input("current-homologation-store", "data"),
    Input("imported-runs-table", "data"),
)
def update_wt_plot(homologation_data, runs_table_data):
    import plotly.graph_objs as go
    import os, json
    if not homologation_data or "reference_folder" not in homologation_data:
        return go.Figure(), "No homologation loaded."
    wt_json_path = os.path.join(homologation_data["reference_folder"], "config", "wt.json")
    if not os.path.exists(wt_json_path):
        return go.Figure(), f"wt.json not found at {wt_json_path}"
    
    # Load run plot config
    run_plot_config_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "config", "run_plot_config.json")
    )
    run_plot_config = {}
    if os.path.exists(run_plot_config_path):
        try:
            with open(run_plot_config_path, "r") as f:
                run_plot_config = json.load(f)
        except Exception:
            pass
    
    # Load brake blanking config
    brake_blanking_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "config", "brake_blanking.json")
    )
    brake_blanking = {}
    if os.path.exists(brake_blanking_path):
        try:
            with open(brake_blanking_path, "r") as f:
                brake_blanking = json.load(f)
        except Exception:
            pass
    
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
        fig = go.Figure(data=go.Scatter(x=cx, y=cz, mode='lines+markers', name='Performance Window', line=dict(color='black'), showlegend=True))
        
        # Track min/max for axis ranges
        x_min, x_max = min(cx), max(cx)
        y_min, y_max = min(cz), max(cz)
        
        # Add imported runs data grouped by run_type
        if runs_table_data:
            # Group runs by run_type (treat homologation and homologation_ref as same group)
            runs_by_type = {}
            for run_row in runs_table_data:
                run_type = run_row.get("run_type", "")
                if not run_type or run_type == "ignore":
                    continue
                    
                try:
                    weighted_cx = float(run_row.get("weighted_Cx", 0))
                    weighted_cz = float(run_row.get("weighted_Cz", 0))
                    offset_cx = float(run_row.get("offset_Cx", 0))
                    offset_cz = float(run_row.get("offset_Cz", 0))
                    
                    # Apply offsets to weighted values
                    final_cx = weighted_cx + offset_cx
                    final_cz = weighted_cz + offset_cz
                    
                    if weighted_cx == 0 and weighted_cz == 0:
                        continue
                    
                    # Normalize run_type: treat homologation_ref as homologation for grouping
                    group_key = "homologation" if run_type == "homologation_ref" else run_type
                    
                    if group_key not in runs_by_type:
                        runs_by_type[group_key] = []
                    runs_by_type[group_key].append({
                        "cx": final_cx,
                        "cz": final_cz,  # Keep original sign, y-axis will be inverted
                        "run": run_row.get("run", ""),
                        "is_ref": run_type == "homologation_ref"  # Track if it needs blanking window
                    })
                    
                    # Update axis ranges
                    x_min = min(x_min, final_cx)
                    x_max = max(x_max, final_cx)
                    y_min = min(y_min, final_cz)
                    y_max = max(y_max, final_cz)
                except (ValueError, TypeError):
                    continue
            
            # Plot each run_type group
            for run_type, run_points in runs_by_type.items():
                # Get plot config for this run type
                config = run_plot_config.get(run_type, {})
                color = config.get("colour", "gray")
                continuous_line = config.get("continuous_line", False)
                
                # Sort by Cx if continuous line
                if continuous_line:
                    run_points = sorted(run_points, key=lambda p: p["cx"])
                
                # Extract Cx and Cz values
                cx_values = [p["cx"] for p in run_points]
                cz_values = [p["cz"] for p in run_points]
                run_names = [p["run"] for p in run_points]
                
                # Determine mode
                mode = 'lines+markers' if continuous_line else 'markers'
                
                # Add trace
                fig.add_trace(go.Scatter(
                    x=cx_values,
                    y=cz_values,
                    mode=mode,
                    name=run_type,
                    line=dict(color=color) if continuous_line else None,
                    marker=dict(color=color, size=8),
                    text=run_names,
                    hovertemplate='<b>%{text}</b><br>Cx: %{x:.4f}<br>Cz: %{y:.4f}<extra></extra>',
                    showlegend=True
                ))
                
                # Add rectangular windows for points marked as reference (is_ref)
                if brake_blanking and any(p.get("is_ref", False) for p in run_points):
                    tolerance = brake_blanking.get("tolerance", {})
                    dx = tolerance.get("dx", 0.01)
                    dz = tolerance.get("dz", 0.035)
                    
                    for point in run_points:
                        # Only draw rectangle if this point is marked as reference
                        if not point.get("is_ref", False):
                            continue
                            
                        cx_pt = point["cx"]
                        cz_pt = point["cz"]
                        
                        # Create rectangle corners
                        rect_x = [cx_pt - dx, cx_pt + dx, cx_pt + dx, cx_pt - dx, cx_pt - dx]
                        rect_y = [cz_pt - dz, cz_pt - dz, cz_pt + dz, cz_pt + dz, cz_pt - dz]
                        
                        # Add filled rectangle with semi-transparent fill
                        # Convert color name to rgba with 20% opacity
                        color_map = {
                            'blue': 'rgba(0, 0, 255, 0.2)',
                            'red': 'rgba(255, 0, 0, 0.2)',
                            'orange': 'rgba(255, 165, 0, 0.2)',
                            'green': 'rgba(0, 128, 0, 0.2)',
                            'gray': 'rgba(128, 128, 128, 0.2)'
                        }
                        fillcolor = color_map.get(color, 'rgba(128, 128, 128, 0.2)')
                        
                        fig.add_trace(go.Scatter(
                            x=rect_x,
                            y=rect_y,
                            mode='lines',
                            fill='toself',
                            fillcolor=fillcolor,
                            line=dict(color=color, width=1),
                            showlegend=False,
                            hoverinfo='skip'
                        ))
        
        # Expand x/y limits by ±0.03
        fig.update_layout(
            xaxis_title="Cx",
            yaxis_title="Cz",
            xaxis_range=[x_min - 0.03, x_max + 0.03],
            yaxis_range=[y_max + 0.03, y_min - 0.03],  # Reversed order to invert y-axis
            legend=dict(x=1.02, y=1, xanchor='left', yanchor='top', orientation='v'),
            margin=dict(r=120),
            autosize=True,
            height=None,
            width=None
        )
        return fig, ""
    except Exception as e:
        return go.Figure(), f"Error reading wt.json: {e}"
