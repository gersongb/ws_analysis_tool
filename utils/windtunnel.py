# Author: Gerson Garsed-Brand
# Date: June 2023
# Description: class definitions for WT activities

import numpy as np
from numpy.matlib import repmat
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math
from typing import Sequence, Optional, NamedTuple, Any
import json
import pickle

# NOTE: The following import was from the original project and is commented out.
# If you need data fitting functionality, implement or adapt it in ws_analysis_tool.utils.
# import aero_characterisation.objects_methods.data_fitting as data_fitting

class WindTunnel:
    def __init__(self, name):
        self.name = name

class Session:
    def __init__(self, name, windtunnel, runs):
        self.name = name
        self.windtunnel = windtunnel
        self.runs = runs

    def add_run(self, run):
        self.runs.append(run)

class Run:
    def __init__(self, number):
        self.number = number

class WeightedData:
    def __init__(self, wt_name, file_path, wt_columns):
        self.wt_name = wt_name
        self.file_path = file_path
        self.wt_columns = wt_columns
        
    def trueround_array(self, array, places=0):
        return np.round(array, places)

    def trueround(self, number, places=0):
        place = 10**(places)
        rounded = (int(number*place + 0.5 if number>=0 else number*place -0.5))/place
        if rounded == int(rounded):
            rounded = int(rounded)
        return rounded

    def extract_data(self):
        columns_original = ['RunNumber', 'RunComment', 'Merit_Czf', 'Merit_Czr', 'Merit_Cz', 'Merit_Cx']
        
        # Get the run number by splitting the parent folder name with the string 'Run' and taking the last element:
        parent_folder = os.path.dirname(self.file_path)            
        run_number = parent_folder.split('Run')[-1]
        
        # Convertion values:
        PSF_to_Pa = 47.88125
        LBF_to_Newtons= 4.44822

        # Create weights dataframe:
        weights_homol = pd.DataFrame()
        weights_homol['name'] = ['First','S01','S02','S03','S04','S05','S06','S07','S08','S09','S10','S11','S12','S13','S14','S15','S16','S17','S18','S19','S20','S21','S22','S23','S24']
        weights_homol['Cz'] = [0,0,0,0,1/15,1/15,1/15,1/15, 1/15, 1/15, 1/15, 1/15, 1/15, 1/15, 1/27, 1/27, 1/27, 1/27, 1/27, 1/27, 1/27, 1/27, 1/27, 0, 0]
        weights_homol['Cx'] = [1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
        
        #weights_scnd = pd.DataFrame()
        #weights_scnd['name'] = ['S01','S02','S03','S04','S05','S06','S07','S08','S09']
        #weights_scnd['Cz'] = [0,0,0.075,0.258333333333333,0.075,0.075,0.258333333333333,0.258333333333333,0]
        #weights_scnd['Cx'] = [1,1,0,0,0,0,0,0,0]
        
        weights_scnd = pd.DataFrame()
        weights_scnd['name'] = ['S01','S04','S08','S09']
        weights_scnd['Cz'] = [0,0.5,0.5,0]
        weights_scnd['Cx'] = [0,0,0,1]

        # Open the file:
        f = open(self.file_path, 'r')

        # Remove the first three rows:
        f.readline()
        f.readline()
        f.readline()

        # Convert the rest of the file to a dataframe:
        df = pd.read_csv(f, sep='\t')

        # Remove the first row:
        df = df.iloc[1:]

        # Retain only colums LF, LR, L, D, DYNPR, RRS Speed Feedback and Air Velocity:
        df = df[['LF', 'LR', 'L', 'D', 'DYNPR', 'RRS Speed Feedback', 'Air Velocity', 'YAW']]
        # Convert all values to float:
        df = df.astype(float)

        # Find the tare values:

        hs_drag_tare_133 = df[(df['RRS Speed Feedback'] > 130) & (df['RRS Speed Feedback'] < 135) & (abs(df['Air Velocity']) < 1)].iloc[-1]


        hs_lift_tare_133 = df[(df['RRS Speed Feedback'] < 5) & (abs(df['Air Velocity']) < 1)].iloc[-1]

        # Create a copy of df and retain only the wind on data:
        df_133 = df[(abs(df['Air Velocity']) > 1) & (abs(df['Air Velocity']) < 150)]

        # Subtract the tare values from the wind on data for all columns:
        results = pd.DataFrame()
        results['Merit_Cx'] = (df_133['D']-hs_drag_tare_133['D'])*LBF_to_Newtons/(df_133['DYNPR']*PSF_to_Pa)
        results['Merit_Cz'] = (df_133['L']-hs_lift_tare_133['L'])*LBF_to_Newtons/(df_133['DYNPR']*PSF_to_Pa)
        results['Merit_Czf'] = (df_133['LF']-hs_lift_tare_133['LF'])*LBF_to_Newtons/(df_133['DYNPR']*PSF_to_Pa)
        results['Merit_Czr'] = (df_133['LR']-hs_lift_tare_133['LR'])*LBF_to_Newtons/(df_133['DYNPR']*PSF_to_Pa)
        results['Air Velocity'] = df_133['Air Velocity']
        results['YAW'] = df_133['YAW']

        # Re-index the dataframe:
        results = results.reset_index(drop=True)

        if len(results) < 15: # short map
            weights = weights_scnd
        else:
            weights = weights_homol

        # Multiply the Cz column by weights['Cz']:
        results['Cz_weighted'] = results['Merit_Cz']*weights['Cz']
        results['Czf_weighted'] = results['Merit_Czf']*weights['Cz']
        results['Czr_weighted'] = results['Merit_Czr']*weights['Cz']

        # Multiply the Cx column by weights['Cx']:
        results['Cx_weighted'] = results['Merit_Cx']*weights['Cx']

        # Removed debug print: results['Cx_weighted']

        SCz = results['Cz_weighted'].sum()
        SCzf = results['Czf_weighted'].sum()
        SCzr = results['Czr_weighted'].sum()

        results['Cx_weighted'] = results['Cx_weighted'].replace(0, np.nan)
        SCx = results['Cx_weighted'].min()  

        tunnel_df = pd.DataFrame()
        tunnel_df['RunNumber'] = [run_number]
        tunnel_df['RunComment'] = ''
        tunnel_df['Merit_Czf'] = round(SCzf, 3)
        tunnel_df['Merit_Czr'] = round(SCzr, 3)
        tunnel_df['Merit_Cz'] = round(SCz, 3)
        tunnel_df['Merit_Cx'] = round(SCx, 3)
        tunnel_df['Merit_Pzf'] = 100* tunnel_df['Merit_Czf'] / tunnel_df['Merit_Cz']
        return tunnel_df

class Aeromap:
    def __init__(self, wt_import_path, run_number, gp_path, poly_path, AAD_default, AAD_correction_min, AAD_correction_max, **kwargs):
        self.wt_import_path = wt_import_path
        self.run_number = run_number
        self.AAD_default = AAD_default
        self.AAD_correction_min = np.array(AAD_correction_min)
        self.AAD_correction_max = np.array(AAD_correction_max)
        self.gp_path = gp_path
        self.poly_path = poly_path
        self.input_names = ['UUTFRh', 'UUTRRh', 'UUTYaw', 'UUTRoll']
        self.output_names = ['Cz', 'Cx', 'Pzf']
        super().__init__(**kwargs)

    def process_wt_data(self):
        pass  # Implement as needed

    def gp_fit(self):
        pass  # Implement as needed

    def save_gp(self, gp):
        pass  # Implement as needed

    def read_gp(self):
        pass  # Implement as needed

    def polynomial_fit(self):
        pass  # Implement as needed

    def save_poly(self, poly):
        pass  # Implement as needed

    def read_poly(self):
        pass  # Implement as needed

    def get_AAD_corrections(self, AAD):
        pass  # Implement as needed
