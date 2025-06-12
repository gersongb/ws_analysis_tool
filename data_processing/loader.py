import pandas as pd

class DataLoader:
    """Class for loading wind tunnel data from various formats."""
    def load_csv(self, filepath):
        return pd.read_csv(filepath)
    # Add more loaders as needed
