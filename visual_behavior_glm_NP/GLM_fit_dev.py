import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from visual_behavior_glm_NP.glm import GLM
import visual_behavior_glm_NP.GLM_params as glm_params
import visual_behavior_glm_NP.GLM_visualization_tools as gvt
import visual_behavior_glm_NP.GLM_analysis_tools as gat
import visual_behavior_glm_NP.GLM_schematic_plots as gsm
import visual_behavior_glm_NP.GLM_fit_tools as gft
from importlib import reload
from alex_utils.alex_utils import *
plt.ion()

if False:
    # Make run JSON
    #####################
    version = '100_testing'
    src_path = '/allen/programs/braintv/workgroups/nc-ophys/alex.piet/NP/visual_behavior_glm_NP/' 
    glm_params.make_run_json(
        version,
        label='testing',
        username='alex', 
        src_path = src_path, 
        TESTING=False
        )

    # Get data information
    cache = glm_params.get_cache()
    experiment_table = glm_params.get_experiment_table()
    unit_table = glm_params.get_unit_table()
    sdk_unit_table = glm_params.get_sdk_unit_table()
    
    # Get model information
    version ='100_testing' 
    run_params = glm_params.load_run_json(version)
    inventory_table = gat.build_inventory_table()

    # Fit results
    oeid = experiment_table.index.values[0]
    session, fit, design = gft.fit_experiment(oeid, run_params)

    # Get dataframes
    results = gat.get_summary_results(version)
    results_pivoted = gat.get_pivoted_results(results)
    weights_df = gat.get_weights_df(version, results_pivoted)
    

def get_analysis_dfs(version):
    run_params = glm_params.load_run_json(version)
    results = gat.get_summary_results(version)
    results_pivoted = gat.get_pivoted_results(results)
    weights_df = gat.get_weights_df(version, results_pivoted)
    return run_params, results, results_pivoted, weights_df


