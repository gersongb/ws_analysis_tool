import dash
from dash import html, dcc
import dash_bootstrap_components as dbc
import dash
import os
import argparse

# Multi-page support
app = dash.Dash(__name__, use_pages=True, external_stylesheets=[dbc.themes.BOOTSTRAP], suppress_callback_exceptions=True)

app.layout = dbc.Container([
    html.H2("Wind Tunnel Analysis Tool", style={"marginTop": 20}),
    dbc.Nav([
        dbc.NavLink("Setup", href="/setup", active="exact"),
        dbc.NavLink("WT Homologation", href="/wt-homologation", active="exact"),
        dbc.NavLink("Live WT Data", href="/live-wt-data", active="exact"),
        dbc.NavLink("Aero Performance", href="/aero-performance", active="exact"),
    ], pills=True, className="mb-3"),
    dash.page_container
], fluid=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Wind Tunnel Analysis Tool')
    parser.add_argument('--port', type=int, default=8050, help='Port to run the app on (default: 8050)')
    parser.add_argument('--host', type=str, default='127.0.0.1', help='Host to run the app on (default: 127.0.0.1)')
    args = parser.parse_args()
    
    app.run(debug=True, port=args.port, host=args.host)
