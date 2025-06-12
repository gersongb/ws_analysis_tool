import dash
from dash import html
import dash_bootstrap_components as dbc

dash.register_page(__name__, path="/wt-homologation", name="WT Homologation")

from dash import dcc, Input, Output, State, ctx

layout = dbc.Container([
    html.H1("WT Homologation"),
    dcc.Store(id="current-homologation-store", storage_type="session"),
    dcc.Store(id="wt-homologation-plot-store"),
    dcc.Graph(id="wt-homologation-plot"),
    html.Div(id="wt-homologation-plot-feedback", style={"marginTop": "10px", "color": "#cc0000"}),
], fluid=True)


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
        fig = go.Figure(data=go.Scatter(x=cx, y=cz, mode='lines+markers', name='WT Path'))
        fig.update_layout(xaxis_title="Cx", yaxis_title="Cz")
        return fig, ""
    except Exception as e:
        return go.Figure(), f"Error reading wt.json: {e}"
