import dash
from dash import html
import dash_bootstrap_components as dbc

dash.register_page(__name__, path="/live-wt-data", name="Live WT Data")

layout = dbc.Container([
    html.H2("Live WT Data"),
    html.P("This is the Live WT Data page.")
], fluid=True)
