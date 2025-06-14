import dash
from dash import html, Output, Input

def register_info_panel_sync_callback(app):
    @app.callback(
        Output("homologation-info-panel", "children"),
        Input("current-homologation-store", "data"),
    )
    def keep_info_panel_in_sync(data):
        if not data:
            return "No homologation loaded."
        return [html.Div([html.B(f"{k}:"), " ", str(v)]) for k, v in data.items()]
