import os
import bz2
import time
import _pickle as cPickle
import xarray as xr
import numpy as np
import pandas as pd
import scipy 
from tqdm import tqdm
from copy import copy
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.linear_model import ElasticNetCV
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import RidgeCV
from sklearn.linear_model import Ridge
from sklearn.linear_model import LassoCV
from sklearn.linear_model import Lasso
from sklearn.linear_model import LassoLarsCV
from sklearn.linear_model import LassoLars
from sklearn.linear_model import SGDRegressor

import visual_behavior_glm_NP.GLM_analysis_tools as gat
import visual_behavior_glm_NP.GLM_params as glm_params

def load_fit_experiment(ecephys_session_id, run_params):
    '''
        Loads the session data, the fit dictionary and the design matrix for this oeid/run_params
    
        Will raise a FileNotFound Exception if the fit did not happen    
    
        INPUTS:
        ecephys_session_id,    oeid for this experiment
        run_params,             dictionary of parameters for the fit version
        
        RETURNS:
        session     SDK session object
        fit         fit dictionary with model results
        design      DesignMatrix object for this experiment
    '''
    fit = gat.load_fit_pkl(run_params, ecephys_session_id)
    session = load_data(ecephys_session_id)
    session = active_passive_split(session,run_params)
    
    # num_weights gets populated during stimulus interpolation
    # configuring it here so the design matrix gets re-generated consistently
    kernels_to_limit_per_image_cycle = ['image0','image1','image2','image3','image4','image5','image6','image7']
    if 'post-omissions' in run_params['kernels']:
        kernels_to_limit_per_image_cycle.append('omissions')
    if 'post-hits' in run_params['kernels']:
        kernels_to_limit_per_image_cycle.append('hits')
        kernels_to_limit_per_image_cycle.append('misses')
        kernels_to_limit_per_image_cycle.append('passive_change')
    for k in kernels_to_limit_per_image_cycle:
        if k in run_params['kernels']:
            run_params['kernels'][k]['num_weights'] = fit['timesteps_per_stimulus']    
    kernels_with_image_cycles = ['hits','misses','passive_change','omissions']
    for k in kernels_with_image_cycles:
        if k in run_params['kernels']:
            num_cycles = int(np.floor(run_params['kernels'][k]['length']/.75))
            run_params['kernels'][k]['num_weights'] = int(fit['timesteps_per_stimulus']*num_cycles)

    design = DesignMatrix(fit)
    design = add_kernels(design, run_params, session,fit)
    design,fit = split_by_engagement(design, run_params, session, fit)
    check_weight_lengths(fit,design)
    return session, fit, design

def check_run_fits(VERSION):
    '''
        Returns the experiment table for this model version with a column 'GLM_fit' 
        appended that is a bool of whether the output pkl file exists for that 
        experiment. It does not try to load the file, or determine if the fit is good.
    
        INPUTS:
        VERSION,    a string of which model version to check
        
        RETURNS
        experiment_table,   a dataframe with a boolean column 'GLM_fit' that says whether VERSION was fit for that experiment id
    '''
    run_params = load_run_json(VERSION)
    experiment_table = pd.read_csv(run_params['experiment_table_path']).reset_index(drop=True).set_index('ecephys_session_id')
    experiment_table['GLM_fit'] = False
    for index, oeid in enumerate(experiment_table.index.values):
        filenamepkl = run_params['experiment_output_dir']+str(oeid)+".pkl" 
        filenamepbz2 = run_params['experiment_output_dir']+str(oeid)+".pbz2" 
        experiment_table.at[oeid, 'GLM_fit'] = os.path.isfile(filenamepkl) or os.path.isfile(filenamepbz2)
    return experiment_table

def check_weight_lengths(fit,design):
    '''
        Does two assertion tests that the number of weights in the design matrix and fit dictionary are 
        consistent with the number of timesteps per stimulus
    '''
    num_weights_per_stimulus = fit['timesteps_per_stimulus']
    num_weights_design = len([x for x in design.get_X().weights.values \
        if x.startswith('image0')])
    assert num_weights_design == num_weights_per_stimulus, \
        "Number of weights in design matrix is incorrect"
    if ('dropouts' in fit) and ('train_weights' in fit['dropouts']['Full']):
        num_weights_fit = len([x for x in \
            fit['dropouts']['Full']['train_weights'].weights.values \
            if x.startswith('image0')])
        assert num_weights_fit == num_weights_per_stimulus, \
            "Number of weights in fit dictionary is incorrect"
    print('Checked weight/kernel lengths against timesteps per stimulus')

def setup_cv(fit,run_params):
    '''
        Defines the time intervals for cross validation
        Two sets of time intervals are generated:
            one for setting the ridge hyperparameter, the other for fitting the model
    '''
    fit['splits'] = split_time(fit['bin_centers'], 
        output_splits=run_params['CV_splits'], 
        subsplits_per_split=run_params['CV_subsplits'])
    fit['ridge_splits'] = split_time(fit['bin_centers'], 
        output_splits=run_params['CV_splits'], 
        subsplits_per_split=run_params['CV_subsplits'])
    return fit

def fit_experiment(oeid, run_params, NO_DROPOUTS=False):
    '''
        Fits the GLM to the ecephys_session_id
        
        Inputs:
        oeid            experiment to fit
        run_params      dictionary of parameters for this fit
        NO_DROPOUTS     if True, does not perform dropout analysis
    
        Returns:
        session         the VBA session object for this experiment
        fit             a dictionary containing the results of the fit
        design          the design matrix for this fit
    '''
    
    # Log oeid
    print("Fitting ecephys_session_id: "+str(oeid)) 
    if run_params['version_type'] == 'production':
        print('Production fit, will include all dropouts')
    elif run_params['version_type'] == 'standard':
        print('Standard fit, will only include standard dropouts')
    elif run_params['version_type'] == 'minimal':
        print('Minimal fit, will not perform dropouts')

    # Warn user if debugging tools are active
    if NO_DROPOUTS:
        print('WARNING! NO_DROPOUTS=True in fit_experiment(), dropout analysis '+\
            'will NOT run, despite version_type')

    # Load Data
    print('Loading data')
    session = load_data(oeid)

    # Processing df/f data
    print('Processing df/f data')
    session, fit, run_params = extract_and_annotate_ephys(session,run_params)

    # Make Design Matrix
    print('Build Design Matrix')
    design = DesignMatrix(fit) 
    design = add_kernels(design, run_params, session, fit) 

    # split by engagement
    design, fit = split_by_engagement(design, run_params, session, fit)

    # Set up CV splits
    print('Setting up CV')
    fit = setup_cv(fit,run_params)

    # Determine Regularization Strength
    print('Evaluating Regularization values')
    fit = evaluate_ridge(fit, design, run_params,session)

    # Set up kernels to drop for model selection
    print('Setting up model selection dropout')
    fit['dropouts'] = copy(run_params['dropouts'])
    if NO_DROPOUTS:
        # Cancel dropouts if we are in debugging mode
        fit['dropouts'] = {'Full':copy(fit['dropouts']['Full'])}
    
    # Iterate over model selections
    print('Iterating over model selection')
    fit = evaluate_models(fit, design, run_params)
    check_weight_lengths(fit,design)

    # Save fit dictionary to compressed pickle file
    print('Saving fit dictionary')
    fit['failed_kernels'] = run_params['failed_kernels']
    fit['failed_dropouts'] = run_params['failed_dropouts']
    filepath = os.path.join(run_params['experiment_output_dir'],str(oeid)+'.pbz2')
    with bz2.BZ2File(filepath, 'w') as f:
        cPickle.dump(fit,f)

    # Pack up
    print('Finished') 
    return session, fit, design

def evaluate_ridge(fit, design,run_params,session):
    '''
        Finds the best L2 value by fitting the model on a grid of L2 values
    
        fit, model dictionary
        design, design matrix
        run_params, dictionary of parameters, which needs to include:
            L2_optimize_by_cell     # If True, uses the best L2 value for each cell
            L2_optimize_by_session  # If True, uses the best L2 value for this session
            L2_use_fixed_value      # If True, uses the hard coded L2_fixed_lambda
            L2_fixed_lambda         # This value is used if L2_use_fixed_value
            L2_grid_range           # Min/Max L2 values for optimization
            L2_grid_num             # Number of L2 values for optimization

        returns fit, with the values added:
            L2_grid                 # the L2 grid evaluated 
            avg_L2_regularization      # the average optimal L2 value, or the fixed value
            cell_L2_regularization     # the optimal L2 value for each cell 
    '''
    if run_params['L2_use_fixed_value']:
        print('Using a hard-coded regularization value')
        fit['avg_L2_regularization'] = run_params['L2_fixed_lambda']
    elif run_params['L2_optimize_by_cre']:
        print('Using a hard-coded regularization value for each cre line')
        this_cre = session.metadata['cre_line']
        fit['avg_L2_regularization'] = run_params['L2_cre_values'][this_cre]
    elif not fit['ok_to_fit_preferred_engagement']:
        print('\tSkipping ridge evaluation because insufficient preferred engagement timepoints')
        fit['avg_L2_regularization'] = np.nan      
        fit['cell_L2_regularization'] = np.empty((fit['spike_count_arr'].shape[1],))
        fit['L2_test_cv'] = np.empty((fit['spike_count_arr'].shape[1],)) 
        fit['L2_train_cv'] = np.empty((fit['spike_count_arr'].shape[1],)) 
        fit['L2_at_grid_min'] = np.empty((fit['spike_count_arr'].shape[1],))
        fit['L2_at_grid_max'] = np.empty((fit['spike_count_arr'].shape[1],))
        fit['cell_L2_regularization'][:] = np.nan
        fit['L2_test_cv'][:] = np.nan
        fit['L2_train_cv'][:] =np.nan
        fit['L2_at_grid_min'][:] =np.nan
        fit['L2_at_grid_max'][:] =np.nan
    elif run_params['ElasticNet']:
        print('Evaluating a grid of regularization values for Lasso')
        fit = evaluate_lasso(fit, design, run_params)
    else:
        print('Evaluating a grid of regularization values')
        if run_params['L2_grid_type'] == 'log':
            fit['L2_grid'] = np.geomspace(run_params['L2_grid_range'][0], \
                run_params['L2_grid_range'][1],num = run_params['L2_grid_num'])
        else:
            fit['L2_grid'] = np.linspace(run_params['L2_grid_range'][0], \
                run_params['L2_grid_range'][1],num = run_params['L2_grid_num'])
        train_cv = np.empty((fit['spike_count_arr'].shape[1], len(fit['L2_grid']))) 
        test_cv  = np.empty((fit['spike_count_arr'].shape[1], len(fit['L2_grid']))) 
        X = design.get_X()
      
        # Iterate over L2 Values 
        for L2_index, L2_value in enumerate(fit['L2_grid']):
            cv_var_train = np.empty((fit['spike_count_arr'].shape[1], len(fit['ridge_splits'])))
            cv_var_test = np.empty((fit['spike_count_arr'].shape[1], len(fit['ridge_splits'])))

            # Iterate over CV splits
            for split_index, test_split in tqdm(enumerate(fit['ridge_splits']), \
                total=len(fit['ridge_splits']), desc='    Fitting L2, {}'.format(L2_value)):
                train_split = np.sort(np.concatenate(\
                    [split for i, split in enumerate(fit['ridge_splits']) if i!=split_index]))
                X_test  = X[test_split,:]
                X_train = X[train_split,:]
                fit_trace_train = fit['spike_count_arr'][train_split,:]
                fit_trace_test  = fit['spike_count_arr'][test_split,:]
                W = fit_regularized(fit_trace_train, X_train, L2_value)
                cv_var_train[:,split_index] = variance_ratio(fit_trace_train, W, X_train)
                cv_var_test[:,split_index]  = variance_ratio(fit_trace_test, W, X_test)

            train_cv[:,L2_index] = np.mean(cv_var_train,1)
            test_cv[:,L2_index]  = np.mean(cv_var_test,1)

        fit['avg_L2_regularization'] = np.mean([fit['L2_grid'][x] for x in np.argmax(test_cv,1)]) 
        fit['cell_L2_regularization'] = [fit['L2_grid'][x] for x in np.argmax(test_cv,1)]     
        fit['L2_test_cv'] = test_cv
        fit['L2_train_cv'] = train_cv
        fit['L2_at_grid_min'] = [x==0 for x in np.argmax(test_cv,1)]
        fit['L2_at_grid_max'] = [x==(len(fit['L2_grid'])-1) for x in np.argmax(test_cv,1)]
    return fit

def evaluate_lasso(fit, design, run_params):
    '''
        Uses Cross validation on a grid of alpha parameters
    '''
    raise Exception('outdated')
    # Determine splits 
    lasso_splits = []  
    for split_index, test_split in enumerate(fit['ridge_splits']):
        train_split = np.sort(np.concatenate(\
            [split for i, split in enumerate(fit['ridge_splits']) if i!=split_index]))
        lasso_splits.append((train_split, test_split)) 

    alphas = [1e-8,1e-7,1e-6,1e-5,1e-4,1e-3,1e-2,1e-1,1,10,100]

    x = design.get_X()
    cell_train_cv = []
    cell_test_cv = []
    for cell_index,cell_value in tqdm(enumerate(fit['fit_trace_arr']['cell_specimen_id'].values),\
        total=len(fit['fit_trace_arr']['cell_specimen_id'].values),desc='   Fitting Cells'):
        y = fit['fit_trace_arr'][:,cell_index] 
        alpha_train_cv = []
        alpha_test_cv = []
        for alpha_dex, alpha_val in enumerate(alphas):
            split_test_ve = []
            split_train_ve = []
            model = LassoLars( #Fill in values
                alpha=alpha_val,
                fit_intercept = False,
                normalize=False,
                max_iter=1000,
                )
            for split_index, split in enumerate(lasso_splits):
                train_y = y[split[0]]
                test_y = y[split[1]]
                train_x = x[split[0],:]
                test_x = x[split[1],:]
                train_y = np.asfortranarray(train_y)
                train_x = np.asfortranarray(train_x)
                model.fit(train_x,train_y)

                split_test_ve.append(model.score(test_x,test_y))
                split_train_ve.append(model.score(train_x,train_y))
            alpha_test_cv.append(np.mean(split_test_ve))
            alpha_train_cv.append(np.mean(split_train_ve))
        cell_train_cv.append(alpha_train_cv)
        cell_test_cv.append(alpha_test_cv)
    fit['Lasso_grid'] = alphas
    fit['avg_Lasso_regularization'] = \
        np.mean([fit['Lasso_grid'][x] for x in np.argmax(cell_test_cv,1)])      
    fit['cell_Lasso_regularization'] = [fit['Lasso_grid'][x] for x in np.argmax(cell_test_cv,1)] 
    fit['Lasso_test_cv'] = cell_test_cv
    fit['Lasso_train_cv'] = cell_train_cv
    fit['Lasso_at_grid_min'] = [x==0 for x in np.argmax(cell_test_cv,1)]
    fit['Lasso_at_grid_max'] = [x==(len(fit['Lasso_grid'])-1) for x in np.argmax(cell_test_cv,1)]
    return fit

def evaluate_models(fit, design, run_params):
    '''
        Evaluates the model selections across all dropouts

    '''
    if not fit['ok_to_fit_preferred_engagement']:
        print('\tSkipping model evaluate because insufficient preferred engagement timepoints')
        cellids = fit['fit_trace_arr']['cell_specimen_id'].values
        W = np.empty((0,fit['fit_trace_arr'].shape[1]))
        W[:] = np.nan
        dummy_weights= xr.DataArray(
            W, 
            dims =('weights','cell_specimen_id'), 
            coords = {  'weights':[], 
                        'cell_specimen_id':cellids}
            )
        fit['dropouts']['Full']['cv_weights']      = \
            np.empty((0,fit['fit_trace_arr'].shape[1], len(fit['splits']))) 
        fit['dropouts']['Full']['cv_var_train']    = \
            np.empty((fit['fit_trace_arr'].shape[1], len(fit['splits'])))
        fit['dropouts']['Full']['cv_var_test']     = \
            np.empty((fit['fit_trace_arr'].shape[1], len(fit['splits'])))
        fit['dropouts']['Full']['cv_adjvar_train'] = \
            np.empty((fit['fit_trace_arr'].shape[1], len(fit['splits'])))
        fit['dropouts']['Full']['cv_adjvar_test']  = \
            np.empty((fit['fit_trace_arr'].shape[1], len(fit['splits'])))
        fit['dropouts']['Full']['cv_adjvar_train_full_comparison'] = \
            np.empty((fit['fit_trace_arr'].shape[1], len(fit['splits'])))
        fit['dropouts']['Full']['cv_adjvar_test_full_comparison']  = \
            np.empty((fit['fit_trace_arr'].shape[1], len(fit['splits'])))
        fit['dropouts']['Full']['train_weights'] = dummy_weights # Needs to be xarray
        fit['dropouts']['Full']['train_variance_explained']    = \
            np.empty((fit['fit_trace_arr'].shape[1],)) 
        fit['dropouts']['Full']['train_adjvariance_explained'] = \
            np.empty((fit['fit_trace_arr'].shape[1],)) 
        fit['dropouts']['Full']['full_model_train_prediction'] = \
            np.empty((0,fit['fit_trace_arr'].shape[1]))
        fit['dropouts']['Full']['cv_weights'][:]      = np.nan
        fit['dropouts']['Full']['cv_var_train'][:]    = np.nan
        fit['dropouts']['Full']['cv_var_test'][:]     = np.nan
        fit['dropouts']['Full']['cv_adjvar_train'][:] = np.nan
        fit['dropouts']['Full']['cv_adjvar_test'][:]  = np.nan
        fit['dropouts']['Full']['cv_adjvar_train_full_comparison'][:] = np.nan
        fit['dropouts']['Full']['cv_adjvar_test_full_comparison'][:]  = np.nan
        fit['dropouts']['Full']['train_variance_explained'][:]    = np.nan 
        fit['dropouts']['Full']['train_adjvariance_explained'][:] = np.nan 
        fit['dropouts']['Full']['full_model_train_prediction'][:] = np.nan  
        return fit
    if run_params['L2_use_fixed_value'] or \
        run_params['L2_optimize_by_session'] or \
        run_params['L2_optimize_by_cre']:
        print('Using a constant regularization value across all cells')
        return evaluate_models_same_ridge(fit,design, run_params)
    elif run_params['L2_optimize_by_cell']:
        print('Using an optimized regularization value for each cell')
        return evaluate_models_different_ridge(fit,design,run_params)
    elif run_params['ElasticNet']:
        print('Using elastic net regularization for each cell')
        return evaluate_models_lasso(fit,design, run_params)
    else:
        raise Exception('Unknown regularization approach')
    print('Done evaluating models')

def evaluate_models_lasso(fit,design,run_params):
    '''
        Fits and evaluates each model defined in fit['dropouts']
           
        For each model, it creates the design matrix, finds the optimal weights, 
        and saves the variance explained. It does this for the entire dataset 
        as test and train. As well as CV, saving each test/train split

    '''
    raise Exception('outdated')
    for model_label in fit['dropouts'].keys():

        # Set up design matrix for this dropout
        X = design.get_X(kernels=fit['dropouts'][model_label]['kernels'])
        mask = get_mask(fit['dropouts'][model_label],design)
        Full_X = design.get_X(kernels=fit['dropouts']['Full']['kernels'])

        # Iterate CV
        cv_var_train    = np.empty((fit['fit_trace_arr'].shape[1], len(fit['splits'])))
        cv_var_test     = np.empty((fit['fit_trace_arr'].shape[1], len(fit['splits'])))
        cv_adjvar_train = np.empty((fit['fit_trace_arr'].shape[1], len(fit['splits']))) 
        cv_adjvar_test  = np.empty((fit['fit_trace_arr'].shape[1], len(fit['splits']))) 
        cv_adjvar_train_fc = np.empty((fit['fit_trace_arr'].shape[1], len(fit['splits']))) 
        cv_adjvar_test_fc= np.empty((fit['fit_trace_arr'].shape[1], len(fit['splits'])))  
        cv_weights      = np.empty((np.shape(X)[1], fit['fit_trace_arr'].shape[1], \
            len(fit['splits'])))
        all_weights     = np.empty((np.shape(X)[1], fit['fit_trace_arr'].shape[1]))
        all_var_explain = np.empty((fit['fit_trace_arr'].shape[1]))
        all_adjvar_explain = np.empty((fit['fit_trace_arr'].shape[1]))
        all_prediction  = np.empty(fit['fit_trace_arr'].shape)
        X_test_array = []   # Cache the intermediate steps for each cell
        X_train_array = []

        for cell_index,cell_value in \
            tqdm(enumerate(fit['fit_trace_arr']['cell_specimen_id'].values),\
            total=len(fit['fit_trace_arr']['cell_specimen_id'].values),desc='   Fitting Cells'):

            fit_trace = fit['fit_trace_arr'][:,cell_index]
            Wall = fit_cell_lasso_regularized(fit_trace, X,\
                fit['cell_Lasso_regularization'][cell_index])     
            var_explain = variance_ratio(fit_trace, Wall,X)
            adjvar_explain = masked_variance_ratio(fit_trace, Wall,X, mask) 
            all_weights[:,cell_index] = Wall
            all_var_explain[cell_index] = var_explain
            all_adjvar_explain[cell_index] = adjvar_explain
            all_prediction[:,cell_index] = X.values @ Wall.values

            for index, test_split in enumerate(fit['splits']):
                train_split = np.sort(np.concatenate(\
                    [split for i, split in enumerate(fit['splits']) if i!=index]))
        
                # If this is the first cell, stash the design matrix and covariance result
                if cell_index == 0:
                    X_test_array.append(X[test_split,:])
                    X_train_array.append(X[train_split,:])
                # Grab the stashed result
                X_test  = X_test_array[index]
                X_train = X_train_array[index]

                fit_trace_train = fit['fit_trace_arr'][train_split,cell_index]
                fit_trace_test = fit['fit_trace_arr'][test_split,cell_index]
                
                # do the fitting 
                W = fit_cell_lasso_regularized(fit_trace_train, X_train, \
                    fit['cell_Lasso_regularization'][cell_index])

                cv_var_train[cell_index,index] = variance_ratio(fit_trace_train, W, X_train)
                cv_var_test[cell_index,index] = variance_ratio(fit_trace_test, W, X_test)
                cv_adjvar_train[cell_index,index]= \
                    masked_variance_ratio(fit_trace_train, W, X_train, mask[train_split]) 
                cv_adjvar_test[cell_index,index] = \
                    masked_variance_ratio(fit_trace_test, W, X_test, mask[test_split])
                cv_weights[:,cell_index,index] = W 
                if model_label == 'Full':
                    # If this is the Full model, the value is the same
                    cv_adjvar_train_fc[cell_index,index]= \
                        masked_variance_ratio(fit_trace_train, W, X_train, mask[train_split])  
                    cv_adjvar_test_fc[cell_index,index] = \
                        masked_variance_ratio(fit_trace_test, W, X_test, mask[test_split])  
                else:
                    # Otherwise, get weights and design matrix for this cell/cv_split 
                    # and compute the variance explained on this mask
                    Full_W = xr.DataArray(fit['dropouts']['Full']['cv_weights'][:,cell_index,index])
                    Full_X_test = Full_X[test_split,:]
                    Full_X_train = Full_X[train_split,:]
                    cv_adjvar_train_fc[cell_index,index]= masked_variance_ratio(\
                        fit_trace_train, Full_W, Full_X_train, mask[train_split])  
                    cv_adjvar_test_fc[cell_index,index] = masked_variance_ratio(\
                        fit_trace_test, Full_W, Full_X_test, mask[test_split])    

        all_weights_xarray = xr.DataArray(
            data = all_weights,
            dims = ("weights", "cell_specimen_id"),
            coords = {
                "weights": X.weights.values,
                "cell_specimen_id": fit['fit_trace_arr'].cell_specimen_id.values
            }
        )

        fit['dropouts'][model_label]['train_weights']   = all_weights_xarray
        fit['dropouts'][model_label]['train_variance_explained']    = all_var_explain
        fit['dropouts'][model_label]['train_adjvariance_explained'] = all_adjvar_explain
        fit['dropouts'][model_label]['full_model_train_prediction'] = all_prediction
        fit['dropouts'][model_label]['cv_weights']      = cv_weights
        fit['dropouts'][model_label]['cv_var_train']    = cv_var_train
        fit['dropouts'][model_label]['cv_var_test']     = cv_var_test
        fit['dropouts'][model_label]['cv_adjvar_train'] = cv_adjvar_train
        fit['dropouts'][model_label]['cv_adjvar_test']  = cv_adjvar_test
        fit['dropouts'][model_label]['cv_adjvar_train_full_comparison'] = cv_adjvar_train_fc
        fit['dropouts'][model_label]['cv_adjvar_test_full_comparison']  = cv_adjvar_test_fc

    return fit 



def evaluate_models_different_ridge(fit,design,run_params):
    '''
        Fits and evaluates each model defined in fit['dropouts']
           
        For each model, it creates the design matrix, finds the optimal weights, 
        and saves the variance explained. It does this for the entire dataset as 
        test and train. As well as CV, saving each test/train split

        Each cell uses a different L2 value defined in fit['cell_L2_regularization']
    '''
    for model_label in fit['dropouts'].keys():

        # Set up design matrix for this dropout
        X = design.get_X(kernels=fit['dropouts'][model_label]['kernels'])
        X_inner = np.dot(X.T, X)
        mask = get_mask(fit['dropouts'][model_label],design)
        Full_X = design.get_X(kernels=fit['dropouts']['Full']['kernels'])

        # Iterate CV
        cv_var_train    = np.empty((fit['fit_trace_arr'].shape[1], len(fit['splits'])))
        cv_var_test     = np.empty((fit['fit_trace_arr'].shape[1], len(fit['splits'])))
        cv_adjvar_train = np.empty((fit['fit_trace_arr'].shape[1], len(fit['splits']))) 
        cv_adjvar_test  = np.empty((fit['fit_trace_arr'].shape[1], len(fit['splits']))) 
        cv_adjvar_train_fc = np.empty((fit['fit_trace_arr'].shape[1], len(fit['splits']))) 
        cv_adjvar_test_fc= np.empty((fit['fit_trace_arr'].shape[1], len(fit['splits'])))  
        cv_weights      = np.empty((np.shape(X)[1], fit['fit_trace_arr'].shape[1], \
            len(fit['splits'])))
        all_weights     = np.empty((np.shape(X)[1], fit['fit_trace_arr'].shape[1]))
        all_var_explain = np.empty((fit['fit_trace_arr'].shape[1]))
        all_adjvar_explain = np.empty((fit['fit_trace_arr'].shape[1]))
        all_prediction  = np.empty(fit['fit_trace_arr'].shape)
        X_test_array = []   # Cache the intermediate steps for each cell
        X_train_array = []
        X_cov_array = []

        for cell_index, cell_value in \
            tqdm(enumerate(fit['fit_trace_arr']['cell_specimen_id'].values),\
            total=len(fit['fit_trace_arr']['cell_specimen_id'].values),desc='   Fitting Cells'):

            fit_trace = fit['fit_trace_arr'][:,cell_index]
            Wall = fit_cell_regularized(X_inner,fit_trace, X,\
                fit['cell_L2_regularization'][cell_index])     
            var_explain = variance_ratio(fit_trace, Wall,X)
            adjvar_explain = masked_variance_ratio(fit_trace, Wall,X, mask) 
            all_weights[:,cell_index] = Wall
            all_var_explain[cell_index] = var_explain
            all_adjvar_explain[cell_index] = adjvar_explain
            all_prediction[:,cell_index] = X.values @ Wall.values

            for index, test_split in enumerate(fit['splits']):
                train_split = np.sort(np.concatenate(\
                    [split for i, split in enumerate(fit['splits']) if i!=index]))
        
                # If this is the first cell, stash the design matrix and covariance result
                if cell_index == 0:
                    X_test_array.append(X[test_split,:])
                    X_train_array.append(X[train_split,:])
                    X_cov_array.append(np.dot(X[train_split,:].T,X[train_split,:]))
                # Grab the stashed result
                X_test  = X_test_array[index]
                X_train = X_train_array[index]
                X_cov   = X_cov_array[index]

                fit_trace_train = fit['fit_trace_arr'][train_split,cell_index]
                fit_trace_test = fit['fit_trace_arr'][test_split,cell_index]
                W = fit_cell_regularized(X_cov,fit_trace_train, X_train, \
                    fit['cell_L2_regularization'][cell_index])
                cv_var_train[cell_index,index] = variance_ratio(fit_trace_train, W, X_train)
                cv_var_test[cell_index,index] = variance_ratio(fit_trace_test, W, X_test)
                cv_adjvar_train[cell_index,index]= masked_variance_ratio(\
                    fit_trace_train, W, X_train, mask[train_split]) 
                cv_adjvar_test[cell_index,index] = masked_variance_ratio(\
                    fit_trace_test, W, X_test, mask[test_split])
                cv_weights[:,cell_index,index] = W 
                if model_label == 'Full':
                    # If this is the Full model, the value is the same
                    cv_adjvar_train_fc[cell_index,index]= masked_variance_ratio(\
                        fit_trace_train, W, X_train, mask[train_split])  
                    cv_adjvar_test_fc[cell_index,index] = masked_variance_ratio(\
                        fit_trace_test, W, X_test, mask[test_split])  
                else:
                    # Otherwise, get weights and design matrix for this cell/cv_split 
                    # and compute the variance explained on this mask
                    Full_W = xr.DataArray(\
                        fit['dropouts']['Full']['cv_weights'][:,cell_index,index])
                    Full_X_test = Full_X[test_split,:]
                    Full_X_train = Full_X[train_split,:]
                    cv_adjvar_train_fc[cell_index,index]= masked_variance_ratio(\
                        fit_trace_train, Full_W, Full_X_train, mask[train_split])  
                    cv_adjvar_test_fc[cell_index,index] = masked_variance_ratio(\
                        fit_trace_test, Full_W, Full_X_test, mask[test_split])    

        all_weights_xarray = xr.DataArray(
            data = all_weights,
            dims = ("weights", "cell_specimen_id"),
            coords = {
                "weights": X.weights.values,
                "cell_specimen_id": fit['fit_trace_arr'].cell_specimen_id.values
            }
        )

        fit['dropouts'][model_label]['train_weights']   = all_weights_xarray
        fit['dropouts'][model_label]['train_variance_explained']    = all_var_explain
        fit['dropouts'][model_label]['train_adjvariance_explained'] = all_adjvar_explain
        #fit['dropouts'][model_label]['full_model_train_prediction'] = all_prediction
        fit['dropouts'][model_label]['cv_weights']      = cv_weights
        fit['dropouts'][model_label]['cv_var_train']    = cv_var_train
        fit['dropouts'][model_label]['cv_var_test']     = cv_var_test
        fit['dropouts'][model_label]['cv_adjvar_train'] = cv_adjvar_train
        fit['dropouts'][model_label]['cv_adjvar_test']  = cv_adjvar_test
        fit['dropouts'][model_label]['cv_adjvar_train_full_comparison'] = cv_adjvar_train_fc
        fit['dropouts'][model_label]['cv_adjvar_test_full_comparison']  = cv_adjvar_test_fc

    return fit 


def evaluate_models_same_ridge(fit, design, run_params):
    '''
        Fits and evaluates each model defined in fit['dropouts']
    
        For each model, it creates the design matrix, finds the optimal weights, 
        and saves the variance explained. It does this for the entire dataset as 
        test and train. As well as CV, saving each test/train split
    
        All cells use the same regularization value defined in fit['avg_L2_regularization']  
        
    '''
    for model_label in fit['dropouts'].keys():

        # Set up design matrix for this dropout
        X = design.get_X(kernels=fit['dropouts'][model_label]['kernels'])
        mask = get_mask(fit['dropouts'][model_label],design)
        Full_X = design.get_X(kernels=fit['dropouts']['Full']['kernels'])

        # Fit on full dataset for references as training fit
        fit_trace = fit['spike_count_arr']
        Wall = fit_regularized(fit_trace, X,fit['avg_L2_regularization'])     
        var_explain = variance_ratio(fit_trace, Wall,X)
        adjvar_explain = masked_variance_ratio(fit_trace, Wall,X, mask) 
        fit['dropouts'][model_label]['train_weights'] = Wall
        fit['dropouts'][model_label]['train_variance_explained']    = var_explain
        fit['dropouts'][model_label]['train_adjvariance_explained'] = adjvar_explain
        #fit['dropouts'][model_label]['full_model_train_prediction'] = X.values @ Wall.values

        # Iterate CV
        cv_var_train = np.empty((fit['spike_count_arr'].shape[1], len(fit['splits'])))
        cv_var_test = np.empty((fit['spike_count_arr'].shape[1], len(fit['splits'])))
        cv_adjvar_train = np.empty((fit['spike_count_arr'].shape[1], len(fit['splits']))) 
        cv_adjvar_test = np.empty((fit['spike_count_arr'].shape[1], len(fit['splits'])))  
        cv_adjvar_train_fc = np.empty((fit['spike_count_arr'].shape[1], len(fit['splits']))) 
        cv_adjvar_test_fc= np.empty((fit['spike_count_arr'].shape[1], len(fit['splits'])))  
        cv_weights = np.empty((np.shape(Wall)[0], np.shape(Wall)[1], len(fit['splits'])))


        for index, test_split in tqdm(enumerate(fit['splits']), total=len(fit['splits']), \
            desc='    Fitting model, {}'.format(model_label)):
            train_split = np.sort(np.concatenate(\
                [split for i, split in enumerate(fit['splits']) if i!=index])) 
            X_test = X[test_split,:]
            X_train = X[train_split,:]
            mask_test = mask[test_split]
            mask_train = mask[train_split]
            fit_trace_train = fit['spike_count_arr'][train_split,:]
            fit_trace_test = fit['spike_count_arr'][test_split,:]
            W = fit_regularized(fit_trace_train, X_train, fit['avg_L2_regularization'])
            cv_var_train[:,index]   = variance_ratio(fit_trace_train, W, X_train)
            cv_var_test[:,index]    = variance_ratio(fit_trace_test, W, X_test)
            cv_adjvar_train[:,index]= masked_variance_ratio(\
                fit_trace_train, W, X_train, mask_train) 
            cv_adjvar_test[:,index] = masked_variance_ratio(\
                fit_trace_test, W, X_test, mask_test)
            cv_weights[:,:,index]   = W 
            if model_label == 'Full':
                # If this model is Full, then the masked variance ratio is the same
                cv_adjvar_train_fc[:,index]= masked_variance_ratio(\
                    fit_trace_train, W, X_train, mask_train)  
                cv_adjvar_test_fc[:,index] = masked_variance_ratio(\
                    fit_trace_test, W, X_test, mask_test)  
            else:
                # Otherwise load the weights and design matrix for this cv_split, 
                # and compute VE with this support mask
                Full_W = xr.DataArray(fit['dropouts']['Full']['cv_weights'][:,:,index])
                Full_X_test = Full_X[test_split,:]
                Full_X_train = Full_X[train_split,:]
                cv_adjvar_train_fc[:,index]= masked_variance_ratio(\
                    fit_trace_train, Full_W, Full_X_train, mask_train)  
                cv_adjvar_test_fc[:,index] = masked_variance_ratio(\
                    fit_trace_test, Full_W, Full_X_test, mask_test)    
        fit['dropouts'][model_label]['cv_weights']      = cv_weights
        fit['dropouts'][model_label]['cv_var_train']    = cv_var_train
        fit['dropouts'][model_label]['cv_var_test']     = cv_var_test
        fit['dropouts'][model_label]['cv_adjvar_train'] = cv_adjvar_train
        fit['dropouts'][model_label]['cv_adjvar_test']  = cv_adjvar_test
        fit['dropouts'][model_label]['cv_adjvar_train_full_comparison'] = cv_adjvar_train_fc
        fit['dropouts'][model_label]['cv_adjvar_test_full_comparison']  = cv_adjvar_test_fc

    return fit 

def get_mask(dropout,design):
    '''
        For the dropout dictionary returns the mask of where the kernels have support in the design matrix.
        Ignores the support of the intercept regressor
    
        INPUTS:
        dropout     a dictionary with keys:
            'is_single' (bool)
            'dropped_kernels' (list)
            'kernels' (list)
        design      DesignMatrix object
    
        RETURNS:
        mask, a boolean vector for the indicies with support

        if the dropout is_single, then support is defined by the included kernels
        if the dropout is not is_single, then support is defined by the dropped kernels
    '''
    if dropout['is_single']:
        # Support is defined by the included kernels
        kernels=dropout['kernels']
    else:
        # Support is defined by the dropped kernels
        kernels=dropout['dropped_kernels']
    
    # Need to remove 'intercept'
    if 'intercept' in kernels:
        kernels.remove('intercept')    

    # Get mask from design matrix object 
    return design.get_mask(kernels=kernels)

def build_dataframe_from_dropouts(fit,run_params):
    '''
        INPUTS:
        threshold (0.005 default) is the minimum amount of variance explained by the full model. The minimum amount of variance explained by a dropout model        

        Returns a dataframe with 
        Index: Cell specimen id
        Columns: Average (across CV folds) variance explained on the test and training sets for each model defined in fit['dropouts']
    '''
        
    cellids = fit['spike_count_arr']['unit_id'].values
    results = pd.DataFrame(index=pd.Index(cellids, name='unit_id'))
    
    # Determines the minimum variance explained for the full model
    if 'dropout_threshold' in run_params:
        threshold = run_params['dropout_threshold']   
    else:
        threshold = 0.005
    
    # Determines whether to record VE and dropout scores without dealing with -Inf on CV splits with no activity
    if 'compute_with_infs' in run_params:
        compute_with_infs = run_params['compute_with_infs']
    else:
        compute_with_infs = True

    # Check for cells with no activity in entire trace
    nan_cells = np.where(np.all(fit['spike_count_arr'] == 0, axis=0))[0]
    if len(nan_cells) > 0:
        print('I found {} cells with all 0 in the spike_count_arr'.format(len(nan_cells)))
        if not run_params['use_events']:
            raise Exception('All 0 in df/f trace')
        else:
            print('Setting Variance Explained to 0')
            fit['nan_cell_ids'] = fit['spike_count_arr'].cell_specimen_id.values[nan_cells]
 
    # Iterate over models
    for model_label in fit['dropouts'].keys():
        # Screen for cells with all 0, and set their variance explained to 0
        if len(nan_cells) > 0:
            fit['dropouts'][model_label]['cv_var_train'][nan_cells,:] = 0     
            fit['dropouts'][model_label]['cv_var_test'][nan_cells,:] = 0     
            fit['dropouts'][model_label]['cv_adjvar_train'][nan_cells,:] = 0     
            fit['dropouts'][model_label]['cv_adjvar_test'][nan_cells,:] = 0     
            fit['dropouts'][model_label]['cv_adjvar_test_full_comparison'][nan_cells,:] = 0     

        # For each model, average over CV splits for variance explained on train/test
        # The CV splits can have NaN for variance explained on the training set if the cell had no detected events in that split
        # Similarly, the test set will have negative infinity. The negative infinities are set to 0 before averaging across CV splits
        # if `compute_with_infs` is True, then a version of the variance explained and dropout scores will be logged where the negative
        # infinities are not set to 0. This is just for debugging purposes.
        results[model_label+"__avg_cv_var_train"] = np.nanmean(fit['dropouts'][model_label]['cv_var_train'],1) 
        if compute_with_infs:
            results[model_label+"__avg_cv_var_train_raw"] = np.mean(fit['dropouts'][model_label]['cv_var_train'],1)
            results[model_label+"__avg_cv_var_test_raw"]  = np.mean(fit['dropouts'][model_label]['cv_var_test'],1) 
            results[model_label+"__avg_cv_var_test_full_comparison_raw"] = np.mean(fit['dropouts']['Full']['cv_var_test'],1)       
 
        # Screen for -Inf values in dropout model
        temp = fit['dropouts'][model_label]['cv_var_test']
        temp[np.isinf(temp)] = 0 
        results[model_label+"__avg_cv_var_test"]  = np.nanmean(temp,1)
        results[model_label+"__avg_cv_var_test_sem"] = np.std(temp,1)/np.sqrt(np.shape(temp)[1])
    
        # Screen for -Inf values in full model
        full_temp = fit['dropouts']['Full']['cv_var_test']
        full_temp[np.isinf(full_temp)] = 0 
        results[model_label+"__avg_cv_var_test_full_comparison"] = np.mean(full_temp,1)

        # For each model, average over CV splits for adjusted variance explained on train/test, and the full model comparison
        # If a CV split did not have an event in a test split, so the kernel has no support, the CV is NAN. Here we use nanmean to
        # ignore those CV splits without information
        results[model_label+"__avg_cv_adjvar_train"] = np.nanmean(fit['dropouts'][model_label]['cv_adjvar_train'],1) 
        if compute_with_infs:
            results[model_label+"__avg_cv_adjvar_test_raw"]  = np.nanmean(fit['dropouts'][model_label]['cv_adjvar_test'],1)
            results[model_label+"__avg_cv_adjvar_test_full_comparison_raw"]  = np.nanmean(fit['dropouts'][model_label]['cv_adjvar_test_full_comparison'],1)
    
        # Screen for -Inf values in dropout model
        temp = fit['dropouts'][model_label]['cv_adjvar_test']
        temp[np.isinf(temp)] = 0 
        results[model_label+"__avg_cv_adjvar_test"]  = np.nanmean(temp,1) 

        # Screen for -Inf values in dropout model
        full_temp = fit['dropouts'][model_label]['cv_adjvar_test_full_comparison']
        full_temp[np.isinf(full_temp)] = 0 
        results[model_label+"__avg_cv_adjvar_test_full_comparison"]  = np.nanmean(full_temp,1) 
        
        # Clip the variance explained values to >= 0
        results.loc[results[model_label+"__avg_cv_var_test"] < 0,model_label+"__avg_cv_var_test"] = 0
        results.loc[results[model_label+"__avg_cv_var_test_full_comparison"] < 0,model_label+"__avg_cv_var_test_full_comparison"] = 0
        results.loc[results[model_label+"__avg_cv_adjvar_test"] < 0,model_label+"__avg_cv_adjvar_test"] = 0
        results.loc[results[model_label+"__avg_cv_adjvar_test_full_comparison"] < 0,model_label+"__avg_cv_adjvar_test_full_comparison"] = 0
        if compute_with_infs:
            results.loc[results[model_label+"__avg_cv_var_test_raw"] < 0,model_label+"__avg_cv_var_test_raw"] = 0
            results.loc[results[model_label+"__avg_cv_var_test_full_comparison_raw"] < 0,model_label+"__avg_cv_var_test_full_comparison_raw"] = 0
            results.loc[results[model_label+"__avg_cv_adjvar_test_raw"] < 0,model_label+"__avg_cv_adjvar_test_raw"] = 0
            results.loc[results[model_label+"__avg_cv_adjvar_test_full_comparison_raw"] < 0,model_label+"__avg_cv_adjvar_test_full_comparison_raw"] = 0

        # Compute the absolute change in variance
        results[model_label+"__absolute_change_from_full"] = results[model_label+"__avg_cv_var_test"] - results[model_label+"__avg_cv_var_test_full_comparison"] 

        # Compute the dropout scores, which is dependent on whether this was a single-dropout or not
        if fit['dropouts'][model_label]['is_single']:  
            # Compute the dropout
            results[model_label+"__dropout"] = -results[model_label+"__avg_cv_var_test"]/results[model_label+"__avg_cv_var_test_full_comparison"]
            results[model_label+"__adj_dropout"] = -results[model_label+"__avg_cv_adjvar_test"]/results[model_label+"__avg_cv_adjvar_test_full_comparison"]
            if compute_with_infs:
                results[model_label+"__adj_dropout_raw"] = -results[model_label+"__avg_cv_adjvar_test_raw"]/results[model_label+"__avg_cv_adjvar_test_full_comparison_raw"]

            # Cleaning Steps, careful eye here! 
            # If the single-dropout explained more variance than the full_comparison, clip dropout to -1
            results.loc[results[model_label+"__avg_cv_adjvar_test_full_comparison"] < results[model_label+"__avg_cv_adjvar_test"], model_label+"__adj_dropout"] = -1 
            results.loc[results[model_label+"__avg_cv_var_test_full_comparison"] < results[model_label+"__avg_cv_var_test"], model_label+"__dropout"] = -1
            if compute_with_infs:
                results.loc[results[model_label+"__avg_cv_adjvar_test_full_comparison_raw"] < results[model_label+"__avg_cv_adjvar_test_raw"], model_label+"__adj_dropout_raw"] = -1 

            # If the single-dropout explained less than THRESHOLD variance, clip dropout to 0            
            results.loc[results[model_label+"__avg_cv_adjvar_test"] < threshold, model_label+"__adj_dropout"] = 0
            results.loc[results[model_label+"__avg_cv_var_test"] < threshold, model_label+"__dropout"] = 0
            if compute_with_infs:
                results.loc[results[model_label+"__avg_cv_adjvar_test_raw"] < threshold, model_label+"__adj_dropout_raw"] = 0
    
            # If the full_comparison model explained less than THRESHOLD variance, clip the dropout to 0.
            results.loc[results[model_label+"__avg_cv_adjvar_test_full_comparison"] < threshold, model_label+"__adj_dropout"] = 0
            results.loc[results[model_label+"__avg_cv_var_test_full_comparison"] < threshold, model_label+"__dropout"] = 0
            if compute_with_infs:
                results.loc[results[model_label+"__avg_cv_adjvar_test_full_comparison_raw"] < threshold, model_label+"__adj_dropout_raw"] = 0
        else:
            # Compute the dropout
            if compute_with_infs:
                results[model_label+"__adj_dropout_raw"] = -(1-results[model_label+"__avg_cv_adjvar_test_raw"]/results[model_label+"__avg_cv_adjvar_test_full_comparison_raw"]) 
            results[model_label+"__adj_dropout"] = -(1-results[model_label+"__avg_cv_adjvar_test"]/results[model_label+"__avg_cv_adjvar_test_full_comparison"]) 
            results[model_label+"__dropout"] = -(1-results[model_label+"__avg_cv_var_test"]/results[model_label+"__avg_cv_var_test_full_comparison"]) 
   
            # Cleaning Steps, careful eye here!             
            # If the dropout explained more variance than the full_comparison, clip the dropout to 0
            if compute_with_infs:
                results.loc[results[model_label+"__avg_cv_adjvar_test_full_comparison_raw"] < results[model_label+"__avg_cv_adjvar_test_raw"], model_label+"__adj_dropout_raw"] = 0
            results.loc[results[model_label+"__avg_cv_adjvar_test_full_comparison"] < results[model_label+"__avg_cv_adjvar_test"], model_label+"__adj_dropout"] = 0
            results.loc[results[model_label+"__avg_cv_var_test_full_comparison"] < results[model_label+"__avg_cv_var_test"], model_label+"__dropout"] = 0

            # If the full_comparison model explained less than THRESHOLD variance, clip the dropout to 0
            if compute_with_infs:
                results.loc[results[model_label+"__avg_cv_adjvar_test_full_comparison_raw"] < threshold, model_label+"__adj_dropout_raw"] = 0
            results.loc[results[model_label+"__avg_cv_adjvar_test_full_comparison"] < threshold, model_label+"__adj_dropout"] = 0
            results.loc[results[model_label+"__avg_cv_var_test_full_comparison"] < threshold, model_label+"__dropout"] = 0

            # fragmentation issues were causing headaches:
            results = results.copy()

    ## Check for NaNs in any column 
    assert results['Full__avg_cv_var_test'].isnull().sum() == 0, "NaNs in variance explained"  
    assert results['Full__avg_cv_var_train'].isnull().sum() == 0, "NaNs in variance explained"  

    if 'var_shuffle_cells' in fit:
        # If the shuffle across cells was computed, record average VE for each cell
        results['Full__shuffle_cells'] = np.nanmean(fit['var_shuffle_cells'],1) 

    if 'var_shuffle_time' in fit:
        # If the shuffle across time was computed, record average VE for each cell
        results['Full__shuffle_time'] = np.nanmean(fit['var_shuffle_time'],1) 

    # Log average regularization value    
    if 'avg_L2_regularization' in fit:
        results['Full__avg_L2_regularization'] = fit['avg_L2_regularization']

    # If cell-wise regularization values were computed, log them
    if 'cell_L2_regularization' in fit:
        results['Full__cell_L2_regularization'] = fit['cell_L2_regularization']   

    return results

def L2_report(fit):
    '''
        Evaluates how well the L2 grid worked. Plots the train/test error across L2 Values to visually see the best value
        Plots the CV_test for each L2 value
    
    '''
    plt.figure()
    plt.plot(fit['L2_grid'], np.mean(fit['L2_train_cv'],0), 'b-')
    plt.plot(fit['L2_grid'], np.mean(fit['L2_test_cv'],0), 'r-')
    plt.gca().set_xscale('log')
    plt.ylabel('Session avg test CV')
    plt.xlabel('L2 Strength')
    plt.axvline(fit['avg_L2_regularization'], color='k', linestyle='--', alpha = 0.5)
    plt.ylim(0,.15) 

    cellids = fit['spike_count_arr']['cell_specimen_id'].values
    results = pd.DataFrame(index=cellids)
    for index, value in enumerate(fit['L2_grid']):
        results["cv_train_"+str(index)] = fit['L2_train_cv'][:,index]
        results["cv_test_"+str(index)]  = fit['L2_test_cv'][:,index]
    results.plot.scatter('cv_test_1','cv_test_'+str(index))
    plt.plot([0,1],[0,1],'k--')
    return results
 
def load_data(oeid):
    '''
        Allen SDK session
        Keyword arguments:
            oeid (int) -- ecephys_session_id
    '''
    cache = glm_params.get_cache()
    session = cache.get_ecephys_session(oeid)
    return session


def process_behavior_predictions(session, ophys_timestamps=None, cutoff_threshold=0.01):
    '''
    Returns a dataframe of licking/grooming behavior derived from behavior videos
    All columns are interpolated onto ophys timestamps
    cutoff_threshold = threshold below which probabilities will be set to 0
    '''
    behavior_predictions = pd.DataFrame({'timestamps':ophys_timestamps})
    for column in ['lick','groom']:
        f = scipy.interpolate.interp1d(
            session.behavior_movie_predictions['timestamps'], 
            session.behavior_movie_predictions[column], 
            bounds_error=False
        )
        behavior_predictions[column] = f(behavior_predictions['timestamps'])
        behavior_predictions[column].fillna(method='ffill',inplace=True)
        # set values below cutoff threshold to 0
        behavior_predictions[column][behavior_predictions[column]<cutoff_threshold] = 0
    return behavior_predictions

def process_eye_data(session,run_params,ophys_timestamps=None):
    '''
        Returns a dataframe of eye tracking data with several processing steps
        1. All columns are interpolated onto ophys timestamps
        2. Likely blinks are removed with a threshold set by run_params['eye_blink_z']
        3. After blink removal, a second transient step removes outliers with threshold run_params['eye_tranisent_threshold']
        4. After interpolating onto the ophys timestamps, Z-scores the eye_width and pupil_radius
        
        Does not modifiy the original eye_tracking dataframe
    '''    

    # Set parameters for blink detection, and load data
    #session.set_params(eye_tracking_z_threshold=run_params['eye_blink_z'])
    eye = session.eye_tracking.copy(deep=True)

    # Compute pupil radius
    eye['pupil_radius'] = np.sqrt(eye['pupil_area']*(1/np.pi))
    
    # Remove likely blinks and interpolate
    eye.loc[eye['likely_blink'],:] = np.nan
    eye = eye.interpolate()   

    # Interpolate everything onto ophys_timestamps
    ophys_eye = pd.DataFrame({'timestamps':ophys_timestamps})
    z_score = ['eye_width','pupil_radius']
    for column in eye.keys():
        if column != 'timestamps':
            f = scipy.interpolate.interp1d(eye['timestamps'], eye[column], bounds_error=False)
            ophys_eye[column] = f(ophys_eye['timestamps'])
            ophys_eye[column].fillna(method='ffill',inplace=True)
            if column in z_score:
                ophys_eye[column+'_zscore'] = scipy.stats.zscore(ophys_eye[column],nan_policy='omit')
    print('                 : '+'mean centering')
    print('                 : '+'standardized to unit variance')
    return ophys_eye 


def extract_and_annotate_ephys(session, run_params):
    '''
        Creates the fit dictionary
        establishes time bins
        processes spike times into spike counts for each time bin
    '''
   
    fit = dict()
    session = active_passive_split(session,run_params)
    fit = establish_ephys_timebins(fit,session, run_params)
    fit = process_ephys_data(fit, session, run_params)

    # If we are splitting on engagement, then determine the engagement timepoints
    if run_params['split_on_engagement']:
        print('Adding Engagement labels. Preferred engagement state: '+\
            run_params['engagement_preference'])
        raise Exception('need to implement engagement splits')
        fit = add_engagement_labels(fit, session, run_params)
    else:
        fit['ok_to_fit_preferred_engagement'] = True
    
    return session, fit, run_params

def active_passive_split(session,run_params):
    # Determine relevant stimuli for this fit
    if run_params['active']:
        print('active session')
        session.filtered_stimulus = \
            session.stimulus_presentations.query('active').copy()
    else:
        print('passive session')
        session.filtered_stimulus = \
            session.stimulus_presentations.query('stimulus_block == 5').copy()   
    session = add_image_index(session)
    return session

def establish_ephys_timebins(fit, session, run_params):
    '''
        Establish the timebins using alot of the same logic as interpolate to stimulus
        make sure timebins don't overlap, etc.
    '''

    # Determine timebins. We start with each stimulus and add time bins 
    # until we hit the next stimulus. Note there are gaps between time bins
    start_times = session.filtered_stimulus.start_time.values
    start_times = np.concatenate([start_times,[start_times[-1]+.75]]) 
    step = run_params['spike_bin_width']
    sets_of_timebin_starts = []
    bins_per_image = int(np.floor(.75/step))-1
    for index, start in enumerate(start_times[0:-1]):
        this_starts = np.arange(start_times[index], start_times[index+1]- step,step)
        sets_of_timebin_starts.append(this_starts[0:bins_per_image]) 

    # Check to make sure we have the same number of timebins per stimulus
    lens = np.array([len(x) for x in sets_of_timebin_starts])
    overlap = np.sum(bins_per_image - lens[np.where(lens !=bins_per_image)[0]])
    if overlap > run_params['image_kernel_overlap_tol']:
        raise Exception('Uneven number of timebins per stimulus: {}'.format(overlap))

    # Make timebins, pack up
    timebin_starts = np.concatenate(sets_of_timebin_starts)
    timebin_ends = timebin_starts + step
    timebins = np.vstack([timebin_starts, timebin_ends]).T 
    fit['timebins'] = timebins
    fit['bin_centers'] = timebin_starts + step/2
    fit['spike_bin_width'] = run_params['spike_bin_width']
    fit['timesteps_per_stimulus'] = bins_per_image
    fit['timebin_vector'] = np.concatenate([fit['timebins'][:,0],[fit['timebins'][-1,1]]])

    # Use the number of timesteps per stimulus to define the image kernel length
    kernels_to_limit_per_image_cycle = \
        ['image0','image1','image2','image3','image4','image5','image6','image7']
    if 'post-omissions' in run_params['kernels']:
        kernels_to_limit_per_image_cycle.append('omissions')
    if 'post-hits' in run_params['kernels']:
        kernels_to_limit_per_image_cycle.append('hits')
        kernels_to_limit_per_image_cycle.append('misses')
        kernels_to_limit_per_image_cycle.append('passive_change')
    for k in kernels_to_limit_per_image_cycle:
        if k in run_params['kernels']:
            run_params['kernels'][k]['num_weights'] = fit['timesteps_per_stimulus']    
    kernels_with_image_cycles = ['hits','misses','passive_change','omissions']
    for k in kernels_with_image_cycles:
        if k in run_params['kernels']:
            num_cycles = int(np.floor(run_params['kernels'][k]['length']/.75))
            run_params['kernels'][k]['num_weights'] = int(fit['timesteps_per_stimulus']*num_cycles)

    return fit

def process_ephys_data(fit, session, run_params):
    # Grab units and spikes from session object
    units = session.get_units()
    channels = session.get_channels()
    unit_channels = units.merge(channels,left_on='peak_channel_id',right_index=True)
    spike_times = session.spike_times
    
    # Filter units
    unit_channels = unit_channels.query(
        '(isi_violations < 0.5) & '+\
        '(amplitude_cutoff < 0.1) & '+\
        '(presence_ratio > 0.95) & '+\
        '(quality == "good")').copy()
 
    # bin spikes 
    spikes = np.zeros([np.shape(fit['timebins'])[0],len(unit_channels)])
    for index, unit in tqdm(enumerate(unit_channels.index.values),
        total = len(unit_channels),desc='binning spikes'):
        spikes[:,index] = get_spike_counts(spike_times[unit],fit['timebins']) 

    # Make xarray
    spike_count_arr = xr.DataArray(
        spikes, 
        dims = ('bin_centers','unit_id'), 
        coords = {
            'bin_centers':fit['bin_centers'],
            'unit_id':unit_channels.index.values
        }
        )
    fit['spike_count_arr'] = spike_count_arr

    # Check to make sure there are no NaNs in the fit_trace
    assert np.isnan(fit['spike_count_arr']).sum() == 0, "Have NaNs in spike_count_arr"

    return fit

def get_spike_counts(spike_times, timebins):
    '''
        spike_times, a list of spike times, sorted
        timebins, numpy array of bins X start/stop
            timebins[i,0] is the start of bin i
            timbins[i,1] is the end of bin i
        

    '''
    counts = np.zeros([np.shape(timebins)[0]])
    spike_pointer =0
    bin_pointer = 0
    while (spike_pointer < len(spike_times))&(bin_pointer < np.shape(timebins)[0]):
        if spike_times[spike_pointer] < timebins[bin_pointer,0]:
            # This spike happens before the time bin, advance spike
            spike_pointer += 1
        elif spike_times[spike_pointer] >= timebins[bin_pointer,1]:
            # This spike happens after the time bin, advance time bin
            bin_pointer += 1
        else:
            counts[bin_pointer] += 1
            spike_pointer += 1
    return counts

def add_rewards_each_flash(session):
    '''
    Append a column to stimulus_presentations which contains the timestamps of rewards that occur
    in a range relative to the onset of the stimulus.

    Args:
        stimulus_presentations (pd.DataFrame): dataframe of stimulus presentations.
            Must contain: 'start_time'
        rewards (pd.DataFrame): rewards dataframe. Must contain 'timestamps'
        range_relative_to_stimulus_start (list with 2 elements): start and end of the range
            relative to the start of each stimulus to average the running speed.
    Returns:
        nothing. session.stimulus_presentations is modified in place with 'rewards' column added
    '''

    rewards_each_flash = add_rewards_each_flash_inner(session.stimulus_presentations,
                                                session.rewards)
    session.stimulus_presentations['rewards'] = rewards_each_flash
    
def add_rewards_each_flash_inner(stimulus_presentations_df, rewards_df,
                       range_relative_to_stimulus_start=[0, 0.75]):
    '''
    Append a column to stimulus_presentations which contains the timestamps of rewards that occur
    in a range relative to the onset of the stimulus.

    Args:
        stimulus_presentations_df (pd.DataFrame): dataframe of stimulus presentations.
            Must contain: 'start_time'
        rewards_df (pd.DataFrame): rewards dataframe. Must contain 'timestamps'
        range_relative_to_stimulus_start (list with 2 elements): start and end of the range
            relative to the start of each stimulus to average the running speed.
    Returns:
        rewards_each_flash (pd.Series): reward times that fell within the window
    '''

    reward_times = rewards_df['timestamps'].values
    stimulus_presentations_df['next_start'] = stimulus_presentations_df['start_time'].shift(-1)
    stimulus_presentations_df.at[stimulus_presentations_df.index[-1], 'next_start'] = \
        stimulus_presentations_df.iloc[-1]['start_time'] + .75
    rewards_each_flash = stimulus_presentations_df.apply(
        lambda row: reward_times[
            ((
                reward_times > row["start_time"]
            ) & (
                reward_times <= row["next_start"]
            ))
        ],
        axis=1,
    )
    stimulus_presentations_df.drop(columns=['next_start'], inplace=True)

    return rewards_each_flash


def check_image_kernel_alignment(design,run_params):
    '''
        Checks to see if any of the image kernels overlap
        Note this fails if the early image is omitted, but I can't check the omission kernels directly because they overlap on purpose with images
    '''
    print('Checking stimulus interpolation')

    kernels = ['image0','image1','image2','image3','image4','image5','image6','image7']
    X = design.get_X(kernels=kernels)
    if ('image_kernel_overlap_tol' in run_params):
        tolerance = run_params['image_kernel_overlap_tol']
    else:
        tolerance = 1
    overlap = np.max(np.sum(X.values, axis=1))
    if overlap > tolerance:
        raise Exception('Image kernels overlap beyond tolerance: {}, {}'.format(overlap, tolerance))
    elif overlap > 1:
        print('Image kernels overlap, but within tolerance: {}, {}'.format(overlap, tolerance))
    else:
        print('No image kernel overlap: {}'.format(overlap))
    print('Passed all interpolation checks')


def add_engagement_labels(fit, session, run_params):
    '''
        Adds a boolean vector 'engaged' to the fit dictionary based on the reward rate 
        The reward rate is determined on an image by image basis by comparing the 
        reward rate to a fixed threshold. Therefore the engagement state only changes 
        at the start/end of each image cycle
    '''

    # Reward rate calculation parameters, hard-coded here
    reward_threshold=1/90
    win_dur=320
    win_type='triang'

    # Get reward rate
    add_rewards_each_flash(session)
    session.stimulus_presentations['rewarded'] = \
        [len(x) > 0 for x in session.stimulus_presentations['rewards']]
    session.stimulus_presentations['reward_rate'] = \
        session.stimulus_presentations['rewarded'].\
        rolling(win_dur,min_periods=1,win_type=win_type).mean()/.75
    session.stimulus_presentations['engaged']= \
        [x > reward_threshold for x in session.stimulus_presentations['reward_rate']]    

    # Make dataframe with start/end of each image cycle pinned with correct engagement value
    start_df = session.stimulus_presentations[['start_time','engaged']].copy()
    end_df = session.stimulus_presentations[['start_time','engaged']].copy()
    end_df['start_time'] = end_df['start_time']+0.75
    engaged_df = pd.concat([start_df,end_df])
    engaged_df = engaged_df.sort_values(by='start_time')\
        .rename(columns={'start_time':'timestamps','engaged':'values'})  
    
    # Interpolate onto fit timestamps
    fit['engaged']= interpolate_to_ophys_timestamps(fit,engaged_df)['values'].values 
    print('\t% of session engaged:    '+str(np.sum(fit['engaged'])/len(fit['engaged'])))
    print('\t% of session disengaged: '+str(1-np.sum(fit['engaged'])/len(fit['engaged'])))

    # Check min_engaged_duration:
    seconds_in_engaged = np.sum(fit['engaged'])/fit['ophys_frame_rate']
    seconds_in_disengaged = np.sum(~fit['engaged'].astype(bool))/fit['ophys_frame_rate']
    if run_params['engagement_preference'] == 'engaged':
        fit['preferred_engagement_state_duration'] = seconds_in_engaged
        fit['ok_to_fit_preferred_engagement'] = \
            seconds_in_engaged > run_params['min_engaged_duration']
    else:
        fit['preferred_engagement_state_duration'] = seconds_in_disengaged
        fit['ok_to_fit_preferred_engagement'] = \
            seconds_in_disengaged > run_params['min_engaged_duration']
    
    if not fit['ok_to_fit_preferred_engagement']:
        print('WARNING, insufficient time points in preferred engagement state.'+\
            ' This model will not fit') 
    return fit

def add_image_index(session):
    images = np.sort(session.filtered_stimulus.image_name.unique())
    image_index = {}
    for index, image in enumerate(images):
        image_index[image]=index
    session.filtered_stimulus['image_index'] = \
        [image_index[x] for x in session.filtered_stimulus['image_name']] 
    return session    

def add_kernels(design, run_params,session, fit):
    '''
        Iterates through the kernels in run_params['kernels'] and adds
        each to the design matrix
        Each kernel must have fields:
            offset:
            length:
    
        design          the design matrix for this model
        run_params      the run_json for this model
        session         the SDK session object for this experiment
        fit             the fit object for this model
    '''
    run_params['failed_kernels']=set()
    run_params['failed_dropouts']=set()
    run_params['kernel_error_dict'] = dict()
    for kernel_name in run_params['kernels']:          
        if 'num_weights' not in run_params['kernels'][kernel_name]:
            run_params['kernels'][kernel_name]['num_weights'] = None
        if run_params['kernels'][kernel_name]['type'] == 'discrete':
            design = add_discrete_kernel_by_label(kernel_name, design, run_params, session, fit)
        else:
            design = add_continuous_kernel_by_label(kernel_name, design, run_params, session, fit)  
    clean_failed_kernels(run_params)
    check_weight_lengths(fit,design)
    check_image_kernel_alignment(design,run_params)

    return design

def clean_failed_kernels(run_params):
    '''
        Modifies the model definition to handle any kernels that failed to fit during 
        the add_kernel process
        Removes the failed kernels from run_params['kernels'], and run_params['dropouts']
    '''
    if run_params['failed_kernels']:
        print('The following kernels failed to be added to the model: ')
        print(run_params['failed_kernels'])
        print()   
 
    # Iterate failed kernels
    for kernel in run_params['failed_kernels']:     
        # Remove the failed kernel from the full list of kernels
        if kernel in run_params['kernels'].keys():
            run_params['kernels'].pop(kernel)

        # Remove the dropout associated with this kernel
        if kernel in run_params['dropouts'].keys():
            run_params['dropouts'].pop(kernel)        
        
        # Remove the failed kernel from each dropout list of kernels
        for dropout in run_params['dropouts'].keys(): 
            # If the failed kernel is in this dropout, remove the kernel from the kernel list
            if kernel in run_params['dropouts'][dropout]['kernels']:
                run_params['dropouts'][dropout]['kernels'].remove(kernel) 
            # If the failed kernel is in the dropped kernel list, remove from dropped kernel list
            if kernel in run_params['dropouts'][dropout]['dropped_kernels']:
                run_params['dropouts'][dropout]['dropped_kernels'].remove(kernel) 

    # Iterate Dropouts, checking for empty dropouts
    drop_list = list(run_params['dropouts'].keys())
    for dropout in drop_list:
        if not (dropout == 'Full'):
            if len(run_params['dropouts'][dropout]['dropped_kernels']) == 0:
                run_params['dropouts'].pop(dropout)
                run_params['failed_dropouts'].add(dropout)
            elif len(run_params['dropouts'][dropout]['kernels']) == 1:
                run_params['dropouts'].pop(dropout)
                run_params['failed_dropouts'].add(dropout)

    if run_params['failed_dropouts']:
        print('The following dropouts failed to be added to the model: ')
        print(run_params['failed_dropouts'])
        print()


def add_continuous_kernel_by_label(kernel_name, design, run_params, session,fit):
    '''
        Adds the kernel specified by <kernel_name> to the design matrix
        kernel_name     <str> the label for this kernel, will raise an error if not implemented
        design          the design matrix for this model
        run_params      the run_json for this model
        session         the SDK session object for this experiment
        fit             the fit object for this model       
    ''' 
    print('    Adding kernel: '+kernel_name)
    try:
        event = run_params['kernels'][kernel_name]['event']

        if not fit['ok_to_fit_preferred_engagement']:
            raise Exception('\tInsufficient time points to add kernel')

        if event == 'intercept':
            timeseries = np.ones(len(fit['bin_centers']))
        elif event == 'time':
            timeseries = np.array(range(1,len(fit['bin_centers'])+1))
            timeseries = timeseries/len(timeseries)
        elif event == 'running':
            running_df = session.running_speed
            running_df = running_df.rename(columns={'speed':'values'})
            timeseries = interpolate_to_ophys_timestamps(fit, running_df)['values'].values
            timeseries = standardize_inputs(timeseries)
        elif event.startswith('face_motion'):
            PC_number = int(event.split('_')[-1])
            face_motion_df =  pd.DataFrame({
                'timestamps': session.behavior_movie_timestamps,
                'values': session.behavior_movie_pc_activations[:,PC_number]
            })
            timeseries = interpolate_to_ophys_timestamps(fit, face_motion_df)['values'].values
            timeseries = standardize_inputs(timeseries, 
                mean_center=run_params['mean_center_inputs'],
                unit_variance=run_params['unit_variance_inputs'])
        elif event == 'population_mean':
            timeseries = np.mean(fit['fit_trace_arr'],1).values
            timeseries = standardize_inputs(timeseries, 
                mean_center=run_params['mean_center_inputs'],
                unit_variance=run_params['unit_variance_inputs'])
        elif event == 'Population_Activity_PC1':
            pca = PCA()
            pca.fit(fit['fit_trace_arr'].values)
            fit_trace_pca = pca.transform(fit['fit_trace_arr'].values)
            timeseries = fit_trace_pca[:,0]
            timeseries = standardize_inputs(timeseries, 
                mean_center=run_params['mean_center_inputs'],
                unit_variance=run_params['unit_variance_inputs'])
        elif (len(event) > 6) & ( event[0:6] == 'model_'):
            bsid = session.metadata['behavior_session_id']
            weight_name = event[6:]
            weight = get_model_weight(bsid, weight_name, run_params)
            weight_df = pd.DataFrame()
            weight_df['timestamps'] = session.stimulus_presentations.start_time.values
            weight_df['values'] = weight.values
            timeseries = interpolate_to_ophys_timestamps(fit, weight_df)
            timeseries['values'].fillna(method='ffill',inplace=True) 
            timeseries = timeseries['values'].values
            timeseries = standardize_inputs(timeseries, 
                mean_center=run_params['mean_center_inputs'],
                unit_variance=run_params['unit_variance_inputs'])
        elif event == 'pupil':
            session.ophys_eye = process_eye_data(session,run_params,
                ophys_timestamps =fit['bin_centers'] )
            timeseries = session.ophys_eye['pupil_radius_zscore'].values
        elif event == 'lick_model' or event == 'groom_model':
            if not hasattr(session, 'lick_groom_model'):
                session.lick_groom_model = process_behavior_predictions(session, 
                    ophys_timestamps = fit['bin_centers'])
            timeseries = session.lick_groom_model[event.split('_')[0]].values
        else:
            raise Exception('Could not resolve kernel label')
    except Exception as e:
        print('\tError encountered while adding kernel for '+kernel_name+'. \n'+\
            '\tAttempting to continue without this kernel.' )
        print(e)
        # Need to remove from relevant lists
        run_params['failed_kernels'].add(kernel_name)      
        run_params['kernel_error_dict'][kernel_name] = {
            'error_type': 'kernel', 
            'kernel_name': kernel_name, 
            'exception':e.args[0], 
            'oeid':session.metadata['ecephys_session_id'], 
            'glm_version':run_params['version']
        }
        return design
    else:
        #assert length of values is same as length of timestamps
        assert len(timeseries) == fit['spike_count_arr'].values.shape[0], \
            'Length of continuous regressor must match length of fit_trace_timestamps'

        # Add to design matrix
        design.add_kernel(
            timeseries, 
            run_params['kernels'][kernel_name]['length'], 
            kernel_name, 
            offset=run_params['kernels'][kernel_name]['offset'],
            num_weights=run_params['kernels'][kernel_name]['num_weights']
        )   
        return design


def standardize_inputs(timeseries, mean_center=True, unit_variance=True,max_value=None):
    '''
        Performs three different input standarizations to the timeseries
    
        if mean_center, the timeseries is adjusted to have 0-mean. This can be performed with unit_variance. 

        if unit_variance, the timeseries is adjusted to have unit variance. This can be performed with mean_center.
    
        if max_value is given, then the timeseries is normalized by max_value. This cannot be performed with mean_center and unit_variance.

    '''
    if (max_value is not None ) & (mean_center or unit_variance):
        raise Exception('Cannot perform max_value standardization and mean_center or unit_variance standardizations together.')

    if mean_center:
        print('                 : '+'mean centering')
        timeseries = timeseries -np.mean(timeseries) # mean center
    if unit_variance:
        print('                 : '+'standardized to unit variance')
        timeseries = timeseries/np.std(timeseries)
    if max_value is not None:
        print('                 : '+'normalized by max value: '+str(max_value))
        timeseries = timeseries/max_value

    return timeseries

def add_discrete_kernel_by_label(kernel_name,design, run_params,session,fit):
    '''
        Adds the kernel specified by <kernel_name> to the design matrix
        kernel_name     <str> the label for this kernel, will raise an error if not implemented
        design          the design matrix for this model
        run_params      the run_json for this model
        session         the SDK session object for this experiment
        fit             the fit object for this model       
    ''' 
    print('    Adding kernel: '+kernel_name)
    try:
        if not fit['ok_to_fit_preferred_engagement']:
            raise Exception('\tInsufficient time points to add kernel') 
        event = run_params['kernels'][kernel_name]['event']
        start_time = fit['timebin_vector'][0]
        end_time = fit['timebin_vector'][-1]
        if event == 'licks':
            event_times = session.licks\
                .query('(timestamps >= @start_time)&(timestamps<@end_time)')['timestamps'].values
            assert run_params['active'], "\tCannot add licks to passive session"
        elif event == 'lick_bouts':
            licks = session.licks
            licks['pre_ILI'] = licks['timestamps'] - licks['timestamps'].shift(fill_value=-10)
            licks['post_ILI'] = licks['timestamps']\
                .shift(periods=-1,fill_value=5000) - licks['timestamps']
            licks['bout_start'] = licks['pre_ILI'] > run_params['lick_bout_ILI']
            licks['bout_end'] = licks['post_ILI'] > run_params['lick_bout_ILI']
            assert np.sum(licks['bout_start']) == np.sum(licks['bout_end']), \
                "Lick bout splitting failed"
            
            # We are making an array of in-lick-bout-event-times by tiling 
            # timepoints every <min_interval> seconds. 
            # If a lick is the end of a bout, the bout-event-times continue 
            # <min_time_per_bout> after the lick
            # Otherwise, we tile the duration of the post_ILI
            event_times = np.concatenate([np.arange(x[0],x[0]\
                +run_params['min_time_per_bout'],run_params['min_interval']) if x[2] else
                np.arange(x[0],x[0]+x[1],run_params['min_interval']) for x in 
                zip(licks['timestamps'], licks['post_ILI'], licks['bout_end'])]) 
        elif event == 'rewards':
            event_times = session.rewards\
                .query('(timestamps >= @start_time)&(timestamps<@end_time)')['timestamps'].values
            assert run_params['active'], "\tCannot add licks to passive session"
        elif event == 'change':
            event_times = session.filtered_stimulus\
                .query('is_change',engine='python')['start_time'].values
        elif event in ['hit', 'miss']:
            if event == 'hit': # Includes auto-rewarded changes as hits 
                event_times = session.filtered_stimulus\
                    .query('is_change & rewarded',engine='python')['start_time'].values 
            elif event == 'miss':
                event_times = session.filtered_stimulus\
                    .query('is_change & ~rewarded',engine='python')['start_time'].values
            event_times = event_times[~np.isnan(event_times)]
            if (len(session.rewards) < 5) or (not run_params['active']):
                raise Exception('\tTrial type regressors arent defined for passive sessions'+\
                    ' (sessions with less than 5 rewards)')
        elif event == 'passive_change':
            if (run_params['active']) or (len(session.rewards) > 5): 
                raise Exception('\tPassive Change kernel cant be added to active sessions')      
            event_times = session.filtered_stimulus\
                .query('is_change',engine='python')['start_time'].values
            event_times = event_times[~np.isnan(event_times)]           
        elif event == 'any-image':
            event_times = session.filtered_stimulus\
                .query('not omitted',engine='python')['start_time'].values
        elif event == 'image_expectation':
            event_times = session.filtered_stimulus['start_time'].values
            # Append last image
            event_times = np.concatenate([event_times,[event_times[-1]+.75]])
        elif event == 'omissions':
            event_times = session.filtered_stimulus\
                .query('omitted',engine='python')['start_time'].values
        elif (len(event)>5) & (event[0:5] == 'image') & ('change' not in event):
            event_times = session.filtered_stimulus\
                .query('image_index == {}'.format(int(event[-1])))['start_time'].values
        elif (len(event)>5) & (event[0:5] == 'image') & ('change' in event):
            event_times = session.filtered_stimulus\
                .query('is_change & (image_index == {})'\
                .format(int(event[-1])))['start_time'].values
        else:
            raise Exception('\tCould not resolve kernel label')

        # Ensure minimum number of events
        if len(event_times) < 5: # HARD CODING THIS VALUE HERE
            raise Exception('\tLess than minimum number of events: '+\
                str(len(event_times)) +' '+event)
    
        # Ensure minimum number of events in preferred engagement state
        check_by_engagement_state(run_params, fit, event_times,event)

    except Exception as e:
        print('\tError encountered while adding kernel for '+kernel_name+'. \n'+\
            '\tAttemping to continue without this kernel.' )
        print(e)
        # Need to remove from relevant lists
        run_params['failed_kernels'].add(kernel_name)      
        run_params['kernel_error_dict'][kernel_name] = {
            'error_type': 'kernel', 
            'kernel_name': kernel_name, 
            'exception':e.args[0], 
            'oeid':session.metadata['ecephys_session_id'], 
            'glm_version':run_params['version']
        }
        return design       
    else:
        events_vec, timestamps = np.histogram(event_times, bins=fit['timebin_vector'])
    
        if (event == 'lick_bouts') or (event == 'licks'): 
            # Force this to be 0 or 1, since we purposefully over-tiled the space. 
            events_vec[events_vec > 1] = 1

        if np.max(events_vec) > 1:
            raise Exception('Had multiple events in the same timebin, {}'.format(kernel_name))

        design.add_kernel(
            events_vec, 
            run_params['kernels'][kernel_name]['length'], 
            kernel_name, 
            offset=run_params['kernels'][kernel_name]['offset'],
            num_weights=run_params['kernels'][kernel_name]['num_weights']
        )   

        return design

def check_by_engagement_state(run_params, fit,event_times,event):
    if not run_params['split_on_engagement']:
        return

    # Bin events onto bin_centers    
    events_vec, timestamps = np.histogram(event_times, bins=fit['fit_trace_bins'])    
    if event == 'lick_bouts': 
        # Force this to be 0 or 1, since we purposefully over-tiled the space. 
        events_vec[events_vec > 1] = 1

    # filter by engagement state
    if run_params['engagement_preference'] == 'engaged': 
        preferred_engagement_state_event_times = events_vec[fit['engaged'].astype(bool)]
        nonpreferred_engagement_state_event_times = events_vec[~fit['engaged'].astype(bool)]
    else:
        preferred_engagement_state_event_times = events_vec[~fit['engaged'].astype(bool)]
        nonpreferred_engagement_state_event_times = events_vec[fit['engaged'].astype(bool)]
    # Check to see if we have enough
    if np.sum(preferred_engagement_state_event_times) < 5:
        raise Exception('\tLess than minimum number of events in preferred engagement state: '+str(np.sum(preferred_engagement_state_event_times)) +' '+event+
        '\n\tTotal number of events: '+str(len(event_times))+
        '\n\tPreferred events:       '+str(np.sum(preferred_engagement_state_event_times))+
        '\n\tNon-Preferred events:   '+str(np.sum(nonpreferred_engagement_state_event_times)))    


class DesignMatrix(object):
    def __init__(self, fit_dict):
        '''
        A toeplitz-matrix builder for running regression with multiple temporal kernels. 

        Args
            fit_dict, a dictionary with:
                event_timestamps: The actual timestamps for each time bin that will be used in the regression model. 
        '''

        # Add some kernels
        self.X = None
        self.kernel_dict = {}
        self.running_stop = 0
        self.events = {'timestamps':fit_dict['bin_centers']}
        self.spike_bin_width=fit_dict['spike_bin_width']

    def make_labels(self, label, num_weights,offset, length): 
        base = [label] * num_weights 
        numbers = [str(x) for x in np.array(range(0,length+1))+offset]
        return [x[0] + '_'+ x[1] for x in zip(base, numbers)]

    def get_mask(self, kernels=None):
        ''' 
            Args:
            kernels, a list of kernel string names
            Returns:
            mask ( a boolean vector), where these kernels have support
        '''
        if len(kernels) == 0:
            X = self.get_X() 
        else:
            X = self.get_X(kernels=kernels) 
        mask = np.any(~(X==0), axis=1)
        return mask.values
    
    def trim_X(self,boolean_mask):
        for kernel in self.kernel_dict.keys():
            self.kernel_dict[kernel]['kernel'] = self.kernel_dict[kernel]['kernel'][:,boolean_mask] 
        for event in self.events.keys():  
            self.events[event] = self.events[event][boolean_mask]
    
    def get_X(self, kernels=None):
        '''
        Get the design matrix. 

        Args:
            kernels (optional list of kernel string names): which kernels to include (for model selection)
        Returns:
            X (np.array): The design matrix
        '''
        if kernels is None:
            kernels = self.kernel_dict.keys()

        kernels_to_use = []
        param_labels = []
        for kernel_name in kernels:
            kernels_to_use.append(self.kernel_dict[kernel_name]['kernel'])
            param_labels.append(self.make_labels(   kernel_name, 
                                                    np.shape(self.kernel_dict[kernel_name]['kernel'])[0], 
                                                    self.kernel_dict[kernel_name]['offset_samples'],
                                                    self.kernel_dict[kernel_name]['kernel_length_samples'] ))

        X = np.vstack(kernels_to_use) 
        x_labels = np.hstack(param_labels)

        assert np.shape(X)[0] == np.shape(x_labels)[0], 'Weight Matrix must have the same length as the weight labels'

        X_array = xr.DataArray(
            X, 
            dims =('weights','timestamps'), 
            coords = {  'weights':x_labels, 
                        'timestamps':self.events['timestamps']}
            )
        return X_array.T

    def add_kernel(self, events, kernel_length, label, offset=0,num_weights=None):
        '''
        Add a temporal kernel. 

        Args:
            events (np.array): The timestamps of each event that the kernel will align to. 
            kernel_length (int): length of the kernel (in SECONDS). 
            label (string): Name of the kernel. 
            offset (int) :offset relative to the events. Negative offsets cause the kernel
                          to overhang before the event (in SECONDS)
        '''
    
        #Enforce unique labels
        if label in self.kernel_dict.keys():
            raise ValueError('Labels must be unique')

        self.events[label] = events

        # CONVERT kernel_length to kernel_length_samples
        if num_weights is None:
            if kernel_length == 0:
                kernel_length_samples = 1
            else:
                kernel_length_samples = int(np.ceil((1/self.spike_bin_width)*kernel_length)) 
        else:
            # Some kernels are hard-coded by number of weights
            kernel_length_samples = num_weights

        # CONVERT offset to offset_samples
        offset_samples = int(np.floor((1/self.spike_bin_width)*offset))

        this_kernel = toeplitz(events, kernel_length_samples, offset_samples)
    
        self.kernel_dict[label] = {
            'kernel':this_kernel,
            'kernel_length_samples':kernel_length_samples,
            'offset_samples':offset_samples,
            'kernel_length_seconds':kernel_length,
            'offset_seconds':offset,
            'ind_start':self.running_stop,
            'ind_stop':self.running_stop+kernel_length_samples
            }
        self.running_stop += kernel_length_samples

def split_by_engagement(design, run_params, session, fit):
    '''
        Splits the elements of fit and design matrix based on the engagement preference
    '''
    
    # If we aren't splitting by engagement, do nothing and return
    if not run_params['split_on_engagement']:
        return design, fit
   
    raise Exception('need to implement') 
    print('Splitting fit dictionary entries by engagement')

    # Set up time arrays, and dff/events arrays to match engagement preference
    fit['engaged_trace_arr'] = fit['fit_trace_arr'][fit['engaged'].astype(bool),:]
    fit['disengaged_trace_arr'] = fit['fit_trace_arr'][~fit['engaged'].astype(bool),:]
    fit['engaged_trace_timestamps'] = fit['bin_centers'][fit['engaged'].astype(bool)]
    fit['disengaged_trace_timestamps'] =fit['bin_centers'][~fit['engaged'].astype(bool)]
    fit['full_trace_arr'] = fit['fit_trace_arr']
    fit['full_trace_timestamps'] = fit['bin_centers']    
    fit['fit_trace_arr'] = fit[run_params['engagement_preference']+'_trace_arr']
    fit['bin_centers'] = fit[run_params['engagement_preference']+'_trace_timestamps']

    # trim design matrix  
    print('Trimming Design Matrix by engagement')
    if run_params['engagement_preference'] == 'engaged':
        design.trim_X(fit['engaged'].astype(bool))
    else:
        design.trim_X(~fit['engaged'].astype(bool)) 
 
    return design,fit

def split_time(timebase, subsplits_per_split=10, output_splits=6):
    '''
        Defines the timepoints for each cross validation split
        timebase        vector for timestamps
        output_splits   number of cross validation splits
        subsplits_per_split     each cv split will be composed of this many continuous blocks

        We have to compose each CV split of a bunch of subsplits to prevent issues  
        time in session from distorting each CV split
    
        returns:
        a list of CV splits. For each list element, a list of the timestamps in that CV split
    '''
    num_timepoints = len(timebase)
    total_splits = output_splits * subsplits_per_split

    # all the split inds broken into the small subsplits
    split_inds = np.array_split(np.arange(num_timepoints), total_splits)

    # make output splits by combining random subsplits
    random_subsplit_inds = np.random.choice(np.arange(total_splits), size=total_splits, replace=False)
    subsplit_inds_per_split = np.array_split(random_subsplit_inds, output_splits)

    output_split_inds = []
    for ind_output in range(output_splits):
        subsplits_this_split = subsplit_inds_per_split[ind_output]
        inds_this_split = np.concatenate([split_inds[sub_ind] for sub_ind in subsplits_this_split])
        output_split_inds.append(inds_this_split)

    output_split_inds = [np.sort(x) for x in output_split_inds]
    return output_split_inds

def toeplitz(events, kernel_length_samples,offset_samples):
    '''
    Build a toeplitz matrix aligned to events.

    Args:
        events (np.array of 1/0): Array with 1 if the event happened at that time, and 0 otherwise.
        kernel_length_samples (int): How many kernel parameters
    Returns
        np.array, size(len(events), kernel_length_samples) of 1/0
    '''

    total_len = len(events)
    events = np.concatenate([events, np.zeros(kernel_length_samples)])
    arrays_list = [events]
    for i in range(kernel_length_samples-1):
        arrays_list.append(np.roll(events, i+1))
    this_kernel= np.vstack(arrays_list)

    #Pad with zeros, roll offset_samples, and truncate to length
    if offset_samples < 0:
        this_kernel = np.concatenate([np.zeros((this_kernel.shape[0], np.abs(offset_samples))), this_kernel], axis=1)
        this_kernel = np.roll(this_kernel, offset_samples)[:, np.abs(offset_samples):]
    elif offset_samples > 0:
        this_kernel = np.concatenate([this_kernel, np.zeros((this_kernel.shape[0], offset_samples))], axis=1)
        this_kernel = np.roll(this_kernel, offset_samples)[:, :-offset_samples]
    return this_kernel[:,:total_len]

def interpolate_to_ophys_timestamps(fit,df):
    '''
    interpolate timeseries onto ophys timestamps

    input:  fit, dictionary containing 'fit_trace_timestamps':<array of timestamps>
            df, dataframe with columns:
                timestamps (timestamps of signal)
                values  (signal of interest)

    returns: dataframe with columns:
                timestamps (fit_trace_timestamps)
                values (values interpolated onto fit_trace_timestamps)
   
    '''
    f = scipy.interpolate.interp1d(
        df['timestamps'],
        df['values'],
        bounds_error=False
    )

    interpolated = pd.DataFrame({
        'timestamps':fit['bin_centers'],
        'values':f(fit['bin_centers'])
    })

    return interpolated

def get_model_weight(bsid, weight_name, run_params):
    '''
        Loads the model weights for <bsid> behavior_ophys_session_id
        Loads only the <weight_name> weight
        run_params gives the directory to the fit location
    '''
    beh_model = pd.read_csv(run_params['beh_model_dir']+str(bsid)+'.csv')
    return beh_model[weight_name].copy()

def fit(fit_trace_arr, X):
    '''
    Analytical OLS solution to linear regression. 

    fit_trace_arr: shape (n_timestamps * n_cells)
    X: shape (n_timestamps * n_kernel_params)
    '''
    W = np.dot(np.linalg.inv(np.dot(X.T, X)), np.dot(X.T, fit_trace_arr))
    return W

def fit_regularized(fit_trace_arr, X, lam):
    '''
    Analytical OLS solution with added L2 regularization penalty. 

    fit_trace_arr: shape (n_timestamps * n_cells)
    X: shape (n_timestamps * n_kernel_params)
    lam (float): Strength of L2 regularization (hyperparameter to tune)

    Returns: XArray
    '''
    # Compute the weights
    if lam == 0:
        W = fit(fit_trace_arr,X)
    else:
        W = np.dot(np.linalg.inv(np.dot(X.T, X) + lam * np.eye(X.shape[-1])),
               np.dot(X.T, fit_trace_arr))

    # Make xarray
    cellids = fit_trace_arr['unit_id'].values
    W_xarray= xr.DataArray(
            W, 
            dims =('weights','unit_id'), 
            coords = {  'weights':X.weights.values, 
                        'unit_id':cellids}
            )
    return W_xarray

def fit_cell_regularized(X_cov,fit_trace_arr, X, lam):
    '''
    Analytical OLS solution with added L2 regularization penalty. 

    fit_trace_arr: shape (n_timestamps * n_cells)
    X: shape (n_timestamps * n_kernel_params)
    lam (float): Strength of L2 regularization (hyperparameter to tune)

    Returns: XArray
    '''
    # Compute the weights
    if lam == 0:
        W = fit(fit_trace_arr,X)
    else:
        W = np.dot(np.linalg.inv(X_cov + lam * np.eye(X.shape[-1])),
               np.dot(X.T, fit_trace_arr))

    # Make xarray 
    W_xarray= xr.DataArray(
            W, 
            dims =('weights'), 
            coords = {  'weights':X.weights.values}
            )
    return W_xarray

def fit_cell_lasso_regularized(fit_trace_arr, X, alpha):
    '''
    Analytical OLS solution with added lasso regularization penalty. 

    fit_trace_arr: shape (n_timestamps * n_cells)
    X: shape (n_timestamps * n_kernel_params)
    alpha (float): Strength of L1 regularization (hyperparameter to tune)

    Returns: XArray
    '''
    # Compute the weights
    if alpha == 0:
        W = fit(fit_trace_arr,X)
    else:
        model = LassoLars(
            alpha=alpha,
            fit_intercept = False,
            normalize=False,
            max_iter=1000,
            )
        model.fit(X,fit_trace_arr)
        W = model.coef_

    # Make xarray 
    W_xarray= xr.DataArray(
            W, 
            dims =('weights'), 
            coords = {  'weights':X.weights.values}
            )
    return W_xarray

def variance_ratio(fit_trace_arr, W, X): 
    '''
    Computes the fraction of variance in fit_trace_arr explained by the linear model Y = X*W
    
    fit_trace_arr: (n_timepoints, n_cells)
    W: Xarray (n_kernel_params, n_cells)
    X: Xarray (n_timepoints, n_kernel_params)
    '''
    #Y = X.values @ W.values
    Y = np.dot(X.values,W.values)
    var_total = np.var(fit_trace_arr, axis=0)   # Total variance in the ophys trace for each cell
    var_resid = np.var(fit_trace_arr-Y, axis=0) # Residual variance in the difference between the model and data
    return (var_total - var_resid) / var_total  # Fraction of variance explained by linear model

def masked_variance_ratio(fit_trace_arr, W, X, mask): 
    '''
    Computes the fraction of variance in fit_trace_arr explained by the linear model Y = X*W
    but only looks at the timepoints in mask
    
    fit_trace_arr: (n_timepoints, n_cells)
    W: Xarray (n_kernel_params, n_cells)
    X: Xarray (n_timepoints, n_kernel_params)
    mask: bool vector (n_timepoints,)
    '''

    Y = np.dot(X.values,W.values)

    # Define variance function that lets us isolate the mask timepoints
    def my_var(trace, support_mask):
        if len(np.shape(trace)) ==1:
            trace = trace.values[:,np.newaxis]
        mu = np.mean(trace,axis=0)
        return np.mean((trace[support_mask,:]-mu)**2,axis=0)

    var_total = my_var(fit_trace_arr, mask)#Total variance in the ophys trace for each cell
    var_resid = my_var(fit_trace_arr-Y, mask)#Residual variance in the difference between the model and data
    return (var_total - var_resid) / var_total  # Fraction of variance explained by linear model

def error_by_time(fit, design):
    '''
        Plots the model error over the course of the session
    '''
    plt.figure()
    Y = design.get_X().values @ fit['dropouts']['Full']['cv_var_weights'][:,:,0]
    diff = fit['fit_trace_arr'] - Y
    plt.figure()
    plt.plot(np.abs(diff.mean(axis=1)), 'k-')
    plt.ylabel('Model Error (df/f)')
    

