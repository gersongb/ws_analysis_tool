import dash
from dash import html
import dash

dash.register_page(__name__, path="/")

layout = html.Div([
    html.H2("Welcome to the Wind Tunnel Data Analysis Dashboard"),
    html.P("Use the navigation to select a page.")
])
