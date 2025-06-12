import dash
from dash import html
import dash

dash.register_page(__name__, path="/analysis")

layout = html.Div([
    html.H3("Data Analysis"),
    html.P("Analysis tools will be available here.")
])
