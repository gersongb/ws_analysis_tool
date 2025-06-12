import plotly.express as px

class PlotFactory:
    """Class for generating plots from wind tunnel data."""
    def scatter(self, df, x, y, color=None, title=None):
        return px.scatter(df, x=x, y=y, color=color, title=title)
    # Add more plotting methods as needed
