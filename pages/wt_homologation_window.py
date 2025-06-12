import dash
from dash import html
import dash_bootstrap_components as dbc

dash.register_page(__name__, path="/wt-homologation-window", name="WT Homologation Window")

from dash import dcc, Input, Output, State, ctx

layout = dbc.Container([
    html.H1("WT Homologation Window"),
    html.P("This is the WT Homologation Window page."),

], fluid=True)

