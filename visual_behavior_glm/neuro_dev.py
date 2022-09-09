import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import psy_output_tools as po
import visual_behavior_glm.PSTH as psth
import visual_behavior_glm.image_regression as ir
import visual_behavior_glm.build_dataframes as bd
import visual_behavior_glm.GLM_fit_dev as gfd
import visual_behavior_glm.GLM_visualization_tools as gvt
import visual_behavior_glm.GLM_analysis_tools as gat
import visual_behavior_glm.GLM_strategy_tools as gst
import visual_behavior_glm.GLM_params as glm_params
from importlib import reload
from alex_utils import *
plt.ion()

'''
Development notes for connecting behavioral strategies and GLM analysis
GLM_strategy_tools.py adds behavioral splits to the kernel regression model
build_dataframes.py generates response_dataframes
PSTH.py plots average population activity using the response_df
image_regression.py analyzes the image by image activity of every cell based on peak_response_df 
'''
## Kernel Regression Analyses
################################################################################

# Get GLM data (Takes a few minutes)
GLM_VERSION = '24_events_all_L2_optimize_by_session'
run_params, results, results_pivoted, weights_df = gfd.get_analysis_dfs(GLM_VERSION)

# get behavior data (fast)
BEHAVIOR_VERSION = 21
summary_df  = po.get_ophys_summary_table(BEHAVIOR_VERSION)
change_df = po.get_change_table(BEHAVIOR_VERSION)
licks_df = po.get_licks_table(BEHAVIOR_VERSION)
bouts_df = po.build_bout_table(licks_df)

# Add behavior information to GLM dataframes
results_beh = gst.add_behavior_session_metrics(results,summary_df)
results_pivoted_beh = gst.add_behavior_session_metrics(results_pivoted,summary_df)
weights_beh = gst.add_behavior_session_metrics(weights_df,summary_df)

# Basic plots
gst.plot_dropout_summary_population(results_beh, run_params)
gst.plot_fraction_summary_population(results_pivoted_beh, run_params)
gst.plot_population_averages(results_pivoted_beh, run_params) 

# Kernel Plots 
gst.compare_cre_kernels(weights_beh, run_params,ym='omissions')
gst.compare_cre_kernels(weights_beh, run_params,ym='omissions',
    compare=['strategy_labels_with_mixed'])
gst.plot_kernels_by_strategy_by_session(weights_beh, run_params,
    ym='omissions', cre_line='Vip-IRES-Cre')
gst.plot_kernels_by_strategy_by_session(weights_beh, run_params,
    ym='omissions', cre_line='Sst-IRES-Cre')
gst.plot_kernels_by_strategy_by_session(weights_beh, run_params,
    ym='omissions', cre_line='Slc17a7-IRES2-Cre')

# Dropout Scatter plots
gst.scatter_by_cell(results_pivoted_beh, run_params)
gst.scatter_by_experience(results_pivoted_beh, run_params, 
    cre_line ='Vip-IRES-Cre',ymetric='omissions')
gst.scatter_dataset(results_pivoted_beh, run_params)


## Generate response dataframes
################################################################################

# Build single session
oeid = summary_df.iloc[0]['ophys_experiment_id'][0]
session = bd.load_data(oeid)
bd.build_response_df_experiment(session)

# Aggregate from hpc results
bd.build_population_df('full_df','Vip-IRES-Cre')

# load finished dataframes
vip_image_df = bd.load_population_df('image_df','Vip-IRES-Cre')
vip_full_df = bd.load_population_df('full_df','Vip-IRES-Cre')


## PSTH - Population average response
################################################################################

# Load each cell type
vip_full_df = bd.load_population_df('full_df','Vip-IRES-Cre')
sst_full_df = bd.load_population_df('full_df','Sst-IRES-Cre')
exc_full_df = bd.load_population_df('full_df','Slc17a7-IRES2-Cre')

# merge cell types
dfs = [exc_full_df, sst_full_df, vip_full_df]
labels =['Excitatory','Sst Inhibitory','Vip Inhibitory']
    
# Plot population response
ax = psth.plot_condition(dfs,'omission',labels)
ax = psth.plot_condition(dfs,'image',labels)
ax = psth.plot_condition(dfs,'change',labels)
ax = psth.plot_condition(dfs,'hit',labels)
ax = psth.plot_condition(dfs,'miss',labels)

# Can split by engagement, generally should plot one strategy at a time
ax = psth.plot_condition(dfs, 'omission',labels,
    split_by_engaged=True,plot_strategy='visual')
ax = psth.plot_condition(dfs, 'omission',labels,
    split_by_engaged=True,plot_strategy='timing')

# Can compare any set of conditions
ax = psth.plot_condition(dfs, ['hit','miss'], labels, plot_strategy='visual')


## Population heatmaps
################################################################################
psth.plot_heatmap(vip_full_df,'Vip', 'omission','Familiar',savefig=True)
psth.plot_heatmap(vip_full_df,'Vip', 'omission','Novel 1',savefig=True)
psth.plot_heatmap(vip_full_df,'Vip', 'omission','Novel >1',savefig=True)

psth.plot_heatmap(sst_full_df,'Sst', 'omission','Familiar',savefig=True)
psth.plot_heatmap(sst_full_df,'Sst', 'omission','Novel 1',savefig=True)
psth.plot_heatmap(sst_full_df,'Sst', 'omission','Novel >1',savefig=True)

psth.plot_heatmap(exc_full_df,'Exc', 'omission','Familiar',savefig=True)
psth.plot_heatmap(exc_full_df,'Exc', 'omission','Novel 1',savefig=True)
psth.plot_heatmap(exc_full_df,'Exc', 'omission','Novel >1',savefig=True)


## QQ Plots 
################################################################################
ax = psth.plot_QQ_strategy(vip_full_df, 'Vip','omission','Familiar')
ax = psth.plot_QQ_engagement(vip_full_df, 'Vip','omission','Familiar')


## Image by Image regression
################################################################################



