import dash
from dash import html
import dash_bootstrap_components as dbc

dash.register_page(__name__, path="/aero-performance", name="Aero Performance")

layout = dbc.Container([
    html.H1("Aero Performance"),
    html.P("This is the Aero Performance page.")
], fluid=True)
