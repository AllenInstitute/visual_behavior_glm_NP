import os 
import json
import shutil
import datetime
import numpy as np
import pandas as pd
from copy import copy
from pathlib import Path

from allensdk.brain_observatory.behavior.behavior_project_cache import \
    VisualBehaviorNeuropixelsProjectCache

OUTPUT_DIR_BASE ='/allen/programs/braintv/workgroups/nc-ophys/alex.piet/NP/ephys/'

def get_versions(vrange=[100,110]):
    versions = os.listdir(OUTPUT_DIR_BASE)
    out_versions = []
    for dex, val in enumerate(np.arange(vrange[0],vrange[1])):
        out_versions = out_versions + [x for x in versions if x.startswith('v_'+str(val)+'_')]
    print('Available GLM model versions')
    for v in out_versions:
        print(v)
    print('')
    return out_versions

def define_kernels():
    kernels = {
        'intercept':    {'event':'intercept',   'type':'continuous',    'length':0,     'offset':0,     'num_weights':None, 'dropout':True, 'text': 'constant value'},
        'hits':         {'event':'hit',         'type':'discrete',      'length':0.75,  'offset':0,     'num_weights':None, 'dropout':True, 'text': 'lick to image change'},
        'misses':       {'event':'miss',        'type':'discrete',      'length':0.75,  'offset':0,     'num_weights':None, 'dropout':True, 'text': 'no lick to image change'},
        'passive_change':   {'event':'passive_change','type':'discrete','length':0.75,  'offset':0,     'num_weights':None, 'dropout':True, 'text': 'passive session image change'},
        'omissions':        {'event':'omissions',   'type':'discrete',  'length':0.75,  'offset':0,     'num_weights':None, 'dropout':True, 'text': 'image was omitted'},
        'each-image':   {'event':'each-image',  'type':'discrete',      'length':0.75,  'offset':0,     'num_weights':None, 'dropout':True, 'text': 'image presentation'},
        'running':      {'event':'running',     'type':'continuous',    'length':1,     'offset':-0.5,  'num_weights':None, 'dropout':True, 'text': 'normalized running speed'},
        'pupil':        {'event':'pupil',       'type':'continuous',    'length':1,     'offset':-0.5,  'num_weights':None, 'dropout':True, 'text': 'Z-scored pupil diameter'},
        'licks':        {'event':'licks',       'type':'discrete',      'length':1,     'offset':-0.5,  'num_weights':None, 'dropout':True, 'text': 'mouse lick'},
    }

    return kernels

def get_cache():
    cache_dir = '/allen/programs/mindscope/workgroups/np-behavior/vbn_data_release/vbn_s3_cache/'
    cache = VisualBehaviorNeuropixelsProjectCache.from_s3_cache(cache_dir=Path(cache_dir))
    return cache

def get_experiment_table():
    '''
        Returns a dataframe of the NP sessions
    '''
    cache = get_cache()
    np_table = cache.get_ecephys_session_table(filter_abnormalities=False)
    np_table = np_table.sort_index()
    return np_table

def get_sdk_unit_table():
    cache = get_cache()
    cell_table = cache.get_unit_table().reset_index()
    
    # A rough filter for good units
    cell_table['good_unit'] =   (cell_table['isi_violations'] < 0.5) &\
                                (cell_table['presence_ratio'] > 0.5) &\
                                (cell_table['amplitude_cutoff']<0.5)
    return cell_table

def get_unit_table():
    '''
        This is an internal unit table with extra columns. I'm going to try to limit how much I 
        use it so things are maximally accessible for external users. 
    '''
    filename = '/allen/programs/mindscope/workgroups/np-behavior/vbn_data_release/supplemental_tables/master_unit_table.csv'
    cell_table = pd.read_csv(filename)

    # A rough filter for good units
    cell_table['good_unit'] =   (cell_table['isi_violations'] < 0.5) &\
                                (cell_table['presence_ratio'] > 0.5)&\
                                (cell_table['amplitude_cutoff']<0.5)&\
                                (cell_table['quality'] == "good")
    return cell_table

def make_run_json(VERSION,label='',username=None, src_path=None, TESTING=False,
    update_version=False):
    '''
        Freezes model files, parameters, and ophys experiment ids
        If the model iteration already exists, throws an error unless (update_version=True)
        root directory is global OUTPUT_DIR_BASE

        v_<VERSION>             contains the model iteration
        ./frozen_model_files/   contains the model files
        ./session_model_files/  contains the output for each session
        ./README.txt            contains information about the model creation
        ./run_params.json       contains a dictionary of the model parameters

        <username>  include a string to README.txt who created each model iteration. 
                    If none is provided attempts to load linux username. 
                    Will default to "unknown" on error
        <label>     include a string to README.txt with a brief summary of the model iteration
        <src_path>  path to repo home. Will throw an error if not passed in
        <TESTING>   if true, will only include 5 sessions in the experiment list
    '''

    # Make directory, will throw an error if already exists

    output_dir              = os.path.join(OUTPUT_DIR_BASE, 'v_'+str(VERSION))
    figure_dir              = os.path.join(output_dir, 'figures')
    fig_coding_dir          = os.path.join(figure_dir, 'coding')
    fig_kernels_dir         = os.path.join(figure_dir, 'kernels')
    model_freeze_dir        = os.path.join(output_dir, 'frozen_model_files')
    experiment_output_dir   = os.path.join(output_dir, 'experiment_model_files')
    manifest_dir            = os.path.join(output_dir, 'manifest')
    manifest                = os.path.join(output_dir, 'manifest', 'manifest.json')
    job_dir                 = os.path.join(output_dir, 'log_files')
    json_path               = os.path.join(output_dir, 'run_params.json')
    experiment_table_path   = os.path.join(output_dir, 'experiment_table_v_'+str(VERSION)+'.csv')

    if not update_version:
        os.mkdir(output_dir)
        os.mkdir(figure_dir)
        os.mkdir(fig_coding_dir)
        os.mkdir(fig_kernels_dir)
        os.mkdir(model_freeze_dir)
        os.mkdir(experiment_output_dir)
        os.mkdir(job_dir)
        os.mkdir(manifest_dir)
    
    # Add a readme file with information about when the model was created
    if not update_version:
        if username is None:
            try:
                username = pwd.getpwuid(os.getuid())[0]
            except:
                username = 'unknown'
        readme_file = os.path.join(output_dir, 'README.txt')
    
        readme = open(readme_file,'w')
        readme.writelines([ 'OPHYS GLM  v',str(VERSION),
                            '\nCreated on ',str(datetime.datetime.now()), 
                            '\nCreated by ',username,
                            '\nComment    ',label,'\n\n'])
    
        readme.close()

    # Copy model files to frozen directory
    python_file_full_path = os.path.join(model_freeze_dir, 'GLM_fit_tools.py')
    python_fit_script = os.path.join(model_freeze_dir, 'fit_glm_v_'+str(VERSION)+'.py')
    if src_path is None:
        raise Exception('You need to provide a path to the model source code')

    shutil.copyfile(os.path.join(src_path, 'visual_behavior_glm_NP/GLM_fit_tools.py'),   python_file_full_path)
    shutil.copyfile(os.path.join(src_path, 'scripts/fit_glm.py'),     python_fit_script)
    

    # Define list of experiments to fit
    experiment_table = get_experiment_table()
    if TESTING:
        experiment_table = experiment_table.tail(5)
    experiment_table.to_csv(experiment_table_path)
    
    # Define job settings
    job_settings = {'queue': 'braintv',
                    'mem': '15g',
                    'walltime': '2:00:00',
                    'ppn':4,
                    }

    # Make JSON file with parameters
    run_params = {
        'output_dir':output_dir,                
        'figure_dir':figure_dir,
        'fig_coding_dir':fig_coding_dir,
        'fig_kernels_dir':fig_kernels_dir,    
        'model_freeze_dir':model_freeze_dir,            
        'experiment_output_dir':experiment_output_dir,
        'job_dir':job_dir,
        'manifest':manifest,
        'json_path':json_path,
        'version':VERSION,
        'creation_time':str(datetime.datetime.now()),
        'user':username,
        'label':label,
        'experiment_table_path':experiment_table_path,
        'src_file':python_file_full_path,
        'fit_script':python_fit_script,
        'ElasticNet': False,            # If True, uses the best L1/L2 value for each session
        'ElasticNet_nalphas':40,        #
        'ElasticNet_nL1_ratios':40,     #
        'L2_optimize_by_cre': False,    # If True, uses the best L2 value for each cell
        'L2_optimize_by_cell': False,    # If True, uses the best L2 value for each cell
        'L2_optimize_by_session': True, # If True, uses the best L2 value for this session
        'L2_use_fixed_value': False,    # If True, uses the hard coded L2_fixed_lambda  
        'L2_fixed_lambda': None,        # This value is used if L2_use_fixed_value   
        'L2_grid_range':[.1, 500],      # Min/Max L2 values for L2_optimize_by_cell, or L2_optimize_by_session
        'L2_grid_num': 40,              # Number of L2 values for L2_optimize_by_cell, or L2_optimize_by_session
        'L2_grid_type':'linear',        # how to space L2 options, must be: 'log' or 'linear'
        'L2_cre_values':{'Slc17a7-IRES2-Cre':340, 'Vip-IRES-Cre':320,'Sst-IRES-Cre':185}, # Fixed values to use for optimize_by_cre 
        'ecephys_session_ids':experiment_table.index.values.tolist(),
        'job_settings':job_settings,
        'split_on_engagement': False,   # If True, uses 'engagement_preference' to determine what engagement state to use
        'engagement_preference': None,  # Either None, "engaged", or "disengaged". Must be None if split_on_engagement is False
        'min_engaged_duration': 600,    # Minimum time, in seconds, the session needs to be in the preferred engagement state 
        'lick_bout_ILI': 0.7,           # The minimum duration of time between two licks to segment them into separate lick bouts
        'min_time_per_bout': 0.2,       # length of bout event that continues after last lick in bout
        'min_interval':0.01,            # over-tiling value for making bout events. Must be << ophys-step-size
        'CV_splits':5,
        'CV_subsplits':10,
        'eye_blink_z': 5.0,             # Threshold for excluding likely blinks
        'eye_transient_threshold': 0.5, # Threshold for excluding eye transients after blink removal
        'mean_center_inputs': True,     # If True, mean centers continuous inputs
        'unit_variance_inputs': True,   # If True, continuous inputs have unit variance
        'max_run_speed': 100,           # If 1, has no effect. Scales running speed to be O(1). 
        'image_kernel_overlap_tol':5,   # Number of timesteps image kernels are allowed to overlap during entire session.
        'dropout_threshold':0.005,      # Minimum variance explained by full model
        'version_type':'standard',      # Should be either 'production' (run everything), 'standard' (run standard dropouts), 'minimal' (just full model)
        'active': True,                # Are we fitting the active behavior (True) or passive (False)
        'spike_bin_width':.025,         # Duration of spike bins in s. Must be cleanly divide 750ms
    } 

    # Define Kernels and dropouts
    kernels = define_kernels()
    kernels = process_kernels(kernels)
    dropouts = define_dropouts(kernels,run_params)
    run_params['kernels']=kernels
    run_params['dropouts']=dropouts

    # Make sub-directories for figure kernels
    for k in kernels:
        os.mkdir(fig_kernels_dir+'/'+k)
    extra_kernels=['all-images','preferred_image','shared_image','non_shared_image']
    for k in extra_kernels:
        os.mkdir(fig_kernels_dir+'/'+k)

    # Regularization parameter checks 
    a = run_params['L2_optimize_by_cell'] 
    b = run_params['L2_optimize_by_session'] 
    c = run_params['L2_use_fixed_value']
    d = run_params['L2_optimize_by_cre']
    e = run_params['ElasticNet']
    assert np.sum([a,b,c,d,e]) == 1, \
        "Must select one and only one regularization option: ElasticNet, L2_optimize_by_cre, L2_optimize_by_cell, L2_optimize_by_session, or L2_use_fixed_value" 

    # Check L2 Fixed value parameters 
    if run_params['L2_use_fixed_value'] and (run_params['L2_fixed_lambda'] is None): 
        raise Exception('L2_use_fixed_value is True, but have None for L2_fixed_lambda') 
    if (not run_params['L2_use_fixed_value']) and (run_params['L2_fixed_lambda'] is not None): 
        raise Exception('L2_use_fixed_value is False, but L2_fixed_lambda has been set')      
    if run_params['L2_use_fixed_value']:
        assert run_params[
                   'L2_fixed_lambda'] > 0, "Must have some positive regularization value to prevent singular matrix"

    # Check L2 Optimization parameters
    if (a or b):
        assert run_params['L2_grid_num'] > 0, "Must have at least one grid option for L2 optimization"
        assert len(run_params['L2_grid_range']) ==2, "Must have a minimum and maximum L2 grid option"
        assert run_params['L2_grid_type'] in ['log','linear'], "L2_grid_type must be log or linear"
        assert run_params['L2_grid_range'][0] > 0, "Must have a positive regularization minimum value."
    
    # Check Engagement split parameters
    if run_params['split_on_engagement']:
        assert (run_params['engagement_preference'] == 'engaged') or (run_params['engagement_preference'] == 'disengaged'), "Splitting on engagement, preference must be 'engaged' or 'disengaged'"
    elif not run_params['split_on_engagement']: 
        assert run_params['engagement_preference'] is None, "Not splitting on engagement, engagement preference must be None"
    assert run_params['min_engaged_duration'] >=0, "Must define a minimum interval for the preferred engagement state"

    # Check version type:
    assert run_params['version_type'] in ['production','standard','minimal'], "Incorrect value for version type. Define what model analyses to run." 

    with open(json_path, 'w') as json_file:
        json.dump(run_params, json_file, indent=4)

    # Print Success
    print('Model Successfully Saved, version ' + str(VERSION))

def process_kernels(kernels):
    '''
        Replaces the 'each-image' kernel with each individual image (not omissions), with the same parameters
    '''
    if ('each-image' in kernels) & ('any-image' in kernels):
        raise Exception('Including both each-image and any-image kernels makes the model unstable')
    if 'each-image' in kernels:
        specs = kernels.pop('each-image')
        for index, val in enumerate(range(0,8)):
            kernels['image'+str(val)] = copy(specs)
            kernels['image'+str(val)]['event'] = 'image'+str(val)
    if 'each-image_change' in kernels:
        specs = kernels.pop('each-image_change')
        for index, val in enumerate(range(0,8)):
            kernels['image_change'+str(val)] = copy(specs)
            kernels['image_change'+str(val)]['event'] = 'image_change'+str(val)
    if 'beh_model' in kernels:
        specs = kernels.pop('beh_model')
        weight_names = ['bias','task0','omissions1','timing1D']
        for index, val in enumerate(weight_names):
            kernels['model_' + str(val)] = copy(specs)
            kernels['model_' + str(val)]['event'] = 'model_' + str(val)
    return kernels

def define_dropouts(kernels,run_params):
    '''
        Creates a dropout dictionary. Each key is the label for the dropout, and the value is a list of kernels to include
        Creates a dropout for each kernel by removing just that kernel.
        Creates a single-dropout for each kernel by removing all but that kernel
        Also defines nested models
    '''

    # Define full model
    dropouts = {'Full': {'kernels':list(kernels.keys()),'dropped_kernels':[],'is_single':False}}

    if run_params['version_type'] in ['production','standard']:
        # Remove each kernel one-by-one
        for kernel in [kernel for kernel in kernels.keys() if kernels[kernel]['dropout']]:
            dropouts[kernel]={'kernels':list(kernels.keys()),'dropped_kernels':[],'is_single':False}
            dropouts[kernel]['kernels'].remove(kernel)
            dropouts[kernel]['dropped_kernels'].append(kernel)

        # Define the nested_models
        dropout_definitions={
            'all-images':           ['image0','image1','image2','image3','image4','image5','image6','image7'],
            'task':                 ['hits','misses','passive_change','post-hits','post-misses','post-passive_change'],
            'behavioral':           ['running','pupil','licks'],
            }
        
        if 'post-omissions' in kernels:
            dropout_definitions['all-omissions'] = ['omissions','post-omissions']
        if 'post-hits' in kernels:
            dropout_definitions['all-hits']=            ['hits','post-hits']
            dropout_definitions['all-misses']=          ['misses','post-misses']
            dropout_definitions['all-passive_change']=  ['passive_change','post-passive_change']
            dropout_definitions['post-task']=           ['post-hits','post-misses','post-passive_change']
            dropout_definitions['task']=                ['hits','misses','passive_change']
            dropout_definitions['all-task']=            ['hits','misses','passive_change','post-hits','post-misses','post-passive_change']
        
        # For each nested model, move the appropriate kernels to the dropped_kernel list
        for dropout_name in dropout_definitions:
            dropouts = set_up_dropouts(dropouts, kernels, dropout_name, dropout_definitions[dropout_name])
    
    # Adds single kernel dropouts:
    if run_params['version_type'] == 'production':
        for drop in [drop for drop in dropouts.keys()]:
            if (drop != 'Full') & (drop != 'intercept'):
                # Make a list of kernels by taking the difference between the kernels in 
                # the full model, and those in the dropout specified by this kernel.
                # This formulation lets us do single kernel dropouts for things like beh_model,
                # or all-images

                kernels = set(dropouts['Full']['kernels'])-set(dropouts[drop]['kernels'])
                kernels.add('intercept') # We always include the intercept
                dropped_kernels = set(dropouts['Full']['kernels']) - kernels
                dropouts['single-'+drop] = {'kernels':list(kernels),'dropped_kernels':list(dropped_kernels),'is_single':True} 
   
    # Check to make sure no kernels got lost in the mix 
    for drop in dropouts.keys():
        assert len(dropouts[drop]['kernels']) + len(dropouts[drop]['dropped_kernels']) == len(dropouts['Full']['kernels']), 'bad length'

    return dropouts

    
def set_up_dropouts(dropouts,kernels,dropout_name, kernel_list):
    '''
        Helper function to define dropouts.
        dropouts,       dictionary of dropout models
        kernels,        dictionary of expanded kernel names
        dropout_name,   name of dropout to be defined
        kernel_list,    list of kernels to be dropped from this nested model
    '''

    dropouts[dropout_name] = {'kernels':list(kernels.keys()),'dropped_kernels':[],'is_single':False}

    for k in kernel_list:
        if k in kernels:
            dropouts[dropout_name]['kernels'].remove(k)
            dropouts[dropout_name]['dropped_kernels'].append(k)
    return dropouts


def load_run_json(version):
    '''
        Loads the run parameters for model v_<version>
        Assumes verion is saved with root directory global OUTPUT_DIR_BASE
        returns a dictionary of run parameters
    '''

    json_path = os.path.join(OUTPUT_DIR_BASE, 'v_'+str(version), 'run_params.json')
    with open(json_path,'r') as json_file:

        run_params = json.load(json_file)

    return run_params


