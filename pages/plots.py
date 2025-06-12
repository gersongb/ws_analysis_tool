import dash
from dash import html
import dash

dash.register_page(__name__, path="/plots")

layout = html.Div([
    html.H3("Data Visualisation"),
    html.P("Plotting tools will be available here.")
])
