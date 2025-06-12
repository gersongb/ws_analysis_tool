import dash
from dash import html, dcc
import dash_bootstrap_components as dbc
import dash
import os

# Multi-page support
app = dash.Dash(__name__, use_pages=True, external_stylesheets=[dbc.themes.BOOTSTRAP])

app.layout = dbc.Container([
    html.H2("Wind Tunnel Analysis Tool", style={"marginTop": 20}),
    dbc.Nav([
        dbc.NavLink("Setup", href="/setup", active="exact"),
        dbc.NavLink("WT Homologation Window", href="/wt-homologation-window", active="exact"),
        dbc.NavLink("Live WT Data", href="/live-wt-data", active="exact"),
        dbc.NavLink("Aero Performance", href="/aero-performance", active="exact"),
    ], pills=True, className="mb-3"),
    dash.page_container
], fluid=True)

if __name__ == "__main__":
    app.run(debug=True)
