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
print(f"DEBUG Layout: run_type_options = {run_type_options}")

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
                                {"name": "", "id": "delete", "presentation": "markdown", "editable": False},
                            ],
                            data=[],
                            editable=True,
                            style_table={"overflowX": "auto"},
                            style_cell={"textAlign": "left", "minWidth": "100px", "maxWidth": "250px", "whiteSpace": "normal"},
                            style_cell_conditional=[
                                {"if": {"column_id": "delete"}, "width": "10px", "minWidth": "10px", "maxWidth": "10px", "textAlign": "center", "padding": "0", "overflow": "hidden"}
                            ],
                            css=[
                                {"selector": ".dash-cell.column-delete", "rule": "cursor: pointer; background: #ffeaea;"}
                            ],
                            style_header={"fontWeight": "bold", "backgroundColor": "#f5f5f5"},
                            page_size=20,
                        )
                    ]
                ),
                
            ], style={"padding": "10px", "border": "2px solid #888", "borderRadius": "8px", "background": "#fafbfc", "overflow": "visible"})
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
)
def update_wt_plot(homologation_data):
    import plotly.graph_objs as go
    import os, json
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
        fig = go.Figure(data=go.Scatter(x=cx, y=cz, mode='lines+markers', name='Performance Window', line=dict(color='black'), showlegend=True))
        # Expand x/y limits by ±0.03
        x_min, x_max = min(cx), max(cx)
        y_min, y_max = min(cz), max(cz)
        fig.update_layout(
            xaxis_title="Cx",
            yaxis_title="Cz",
            xaxis_range=[x_min - 0.03, x_max + 0.03],
            yaxis_range=[y_min - 0.03, y_max + 0.03],
            legend=dict(x=1.02, y=1, xanchor='left', yanchor='top', orientation='v'),
            margin=dict(r=120),
            autosize=True,
            height=None,
            width=None
        )
        return fig, ""
    except Exception as e:
        return go.Figure(), f"Error reading wt.json: {e}"
