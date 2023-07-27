import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from visual_behavior_glm_NP.glm import GLM
import visual_behavior_glm_NP.GLM_params as glm_params
import visual_behavior_glm_NP.GLM_visualization_tools as gvt
import visual_behavior_glm_NP.GLM_analysis_tools as gat
import visual_behavior_glm_NP.GLM_schematic_plots as gsm
import visual_behavior_glm_NP.GLM_fit_tools as gft
import visual_behavior_glm_NP.GLM_cell_metrics as gcm
from importlib import reload
from alex_utils.alex_utils import *
plt.ion()

if False:
    # Make run JSON
    #####################
    version = '101_testing_active'
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
    version ='100_testing_active' 
    run_params = glm_params.load_run_json(version)
    inventory_table = gat.build_inventory_table()

    # Fit results
    oeid = experiment_table.index.values[0]
    session, fit, design = gft.fit_experiment(oeid, run_params)

    # Get dataframes
    results = gat.get_summary_results(version)
    results_pivoted = gat.get_pivoted_results(results)
    weights_df = gat.get_weights_df(version, results_pivoted)
    
    # Get everything at once
    run_params, results, results_pivoted, weights_df = gfd.get_analysis_dfs(version)

    # Merge active/passive
    #df_combined = gat.merge_active_passive(df_active, df_passive)
    results_c = gat.merge_active_passive(results_a, results_p)
    results_pivoted_c = gat.merge_active_passive(results_pivoted_a, results_pivoted_p)
    weights_df_c = gat.merge_active_passive(weights_df_a, weights_df_p)
    stats = gvt.var_explained_by_experience(results_pivoted_c, run_params,merged=True) 
    stats = gvt.plot_dropout_summary_population(results_c, run_params,merged=True) 
    stats = gvt.plot_dropout_summary_by_area(results_c, run_params, 'all-images',merged=True)

    # Evaluate model fit quality
    stats = gvt.var_explained_by_experience(results_pivoted, run_params)
  
    # Get boxplot of coding scores by experience 
    stats = gvt.plot_dropout_summary_population(results, run_params) 
    stats = gvt.plot_dropout_summary_by_area(results, run_params, 'all-images')
    stats = gvt.plot_dropout_summary_by_area(results, run_params, 'omissions')
    stats = gvt.plot_dropout_summary_by_area(results, run_params, 'behavioral')
    stats = gvt.plot_dropout_summary_by_area(results, run_params, 'task')
    
    # Kernel plots
    gvt.kernel_evaluation(weights_df, run_params, 'omissions',
        session_filter=['Familiar'])
    gvt.kernel_evaluation(weights_df, run_params, 'omissions',
        area_filter=['CA1'])
    gvt.plot_kernel_comparison(weights_df, run_params, 'omissions') 
    gvt.plot_kernel_comparison(weights_df, run_params, 'all-images') 
    gvt.plot_kernel_comparison(weights_df, run_params, 'hits') 
    gvt.plot_kernel_comparison(weights_df, run_params, 'misses') 
    gvt.plot_kernel_comparison(weights_df, run_params, 'licks') 
    gvt.plot_kernel_comparison(weights_df, run_params, 'running') 
    gvt.plot_kernel_comparison(weights_df, run_params, 'pupil') 

    # Shared/Non Shared images
    gvt.plot_kernel_comparison(weights_df, run_params, 'shared_image')
    gvt.plot_kernel_comparison(weights_df, run_params, 'non_shared_image')
    gvt.plot_kernel_comparison_shared_images(weights_df,run_params)

def get_analysis_dfs(version):
    run_params = glm_params.load_run_json(version)
    results = gat.get_summary_results(version)
    results_pivoted = gat.get_pivoted_results(results)
    weights_df = gat.get_weights_df(version, results_pivoted)
    return run_params, results, results_pivoted, weights_df

def figure_dump(version, run_params, results, results_pivoted, weights_df):
    stats = gvt.var_explained_by_experience(results_pivoted, run_params,savefig=True)
    stats = gvt.plot_dropout_summary_population(results, run_params,savefig=True)
    dropouts = ['all-images','omissions','behavioral','pupil','running','licks',
        'task','hits','misses']
    for d in dropouts:
        stats = gvt.plot_dropout_summary_by_area(results, run_params,d,savefig=True)
    closeall()

    kernels = ['omissions','all-images','hits','misses','licks','running','pupil',
        'shared_image','non_shared_image']
    areas = weights_df['structure_acronym'].unique()
    for k in kernels:
        gvt.plot_kernel_comparison(weights_df, run_params, k)
        for a in areas:
            try:
                gvt.plot_kernel_comparison(weights_df, run_params, k,area_filter=[a])
            except:
                pass
            closeall()

    for k in kernels:
        for a in areas:
            try:
                gvt.kernel_evaluation(weights_df, run_params, k,session_filter=['Familiar'], 
                    area_filter=[a],save_results=True)
                gvt.kernel_evaluation(weights_df, run_params, k,session_filter=['Novel'], 
                    area_filter=[a],save_results=True)
            except:
                pass
            closeall()



 
