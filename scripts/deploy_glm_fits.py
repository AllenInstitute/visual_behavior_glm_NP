import os
import argparse
import sys
import time
import pandas as pd
import numpy as  np
from simple_slurm import Slurm

import visual_behavior.database as db
import visual_behavior_glm.GLM_params as glm_params

parser = argparse.ArgumentParser(description='deploy glm fits to cluster')
parser.add_argument('--env-path', type=str, default='visual_behavior', metavar='path to conda environment to use')
parser.add_argument('--version', type=str, default='0', metavar='glm version')
parser.add_argument(
    '--src-path', 
    type=str, 
    default='',
    metavar='src_path',
    help='folder where code lives'
)
parser.add_argument(
    '--force-overwrite', 
    action='store_true',
    default=False,
    dest='force_overwrite', 
    help='Overwrites existing fits for this version if enabled. Otherwise only experiments without existing results are fit'
)

parser.add_argument(
    '--use-previous-fit', 
    action='store_true',
    default=False,
    dest='use_previous_fit', 
    help='use previous fit if it exists (boolean, default = False)'
)
parser.add_argument(
    '--job-start-fraction', 
    type=float, 
    default=0.0,
    metavar='start_fraction',
    help='which fraction of all jobs to start on. useful if splitting jobs amongst users. Default = 0.0'
)
parser.add_argument(
    '--job-end-fraction', 
    type=float, 
    default=1.0,
    metavar='end_fraction',
    help='which fraction of all jobs to end on. useful if splitting jobs amongst users. Default = 1.0'
)

def calculate_required_mem(unit_count):
    '''calculate required memory in GB'''
    return 12 + 0.25*unit_count

def calculate_required_walltime(unit_count):
    '''calculate required walltime in hours'''
    estimate= 10 + 0.125*unit_count
    return np.min([estimate,48]) 

def get_unit_count(ecephys_session_id):
    '''
        get number of units for each session
    '''
    unit_table = glm_params.get_unit_table()
    return len(unit_table.query('ecephys_session_id == @ecephys_session_id'))

def already_fit(oeid, version):
    '''
    check the weight_matrix_lookup_table to see if an oeid/glm_version combination has already been fit
    returns a boolean
    '''
    conn = db.Database('visual_behavior_data')
    coll = conn['ophys_glm']['weight_matrix_lookup_table']
    document_count = coll.count_documents({'ecephys_session_id':int(oeid), 'glm_version':str(version)})
    conn.close()
    return document_count > 0

if __name__ == "__main__":
    args = parser.parse_args()
    python_executable = "{}/bin/python".format(args.env_path)
    print('python executable = {}'.format(python_executable))
    python_file = "{}/scripts/fit_glm.py".format(args.src_path)

    stdout_basedir = "/allen/programs/braintv/workgroups/nc-ophys/alex.piet/NP/ephys/logs"
    stdout_location = os.path.join(stdout_basedir, 'job_records_{}'.format(args.version))
    if not os.path.exists(stdout_location):
        print('making folder {}'.format(stdout_location))
        os.mkdir(stdout_location)
    print('stdout files will be at {}'.format(stdout_location))

    experiment_table = glm_params.get_experiment_table().reset_index()
    run_params = glm_params.load_run_json(args.version)
    print('experiments table loaded')

    # get ROI count for each experiment
    experiments_table['unit_count'] = experiments_table['ecephys_session_id'].map(lambda oeid: get_unit_count(oeid))
    print('unit counts extracted')

    job_count = 0

    if args.use_previous_fit:
        job_string = "--oeid {} --version {} --use-previous-fit"
    else:
        job_string = "--oeid {} --version {}"

    experiment_ids = experiments_table['ecephys_session_id'].values
    n_experiment_ids = len(experiment_ids)

    for experiment_id in experiment_ids[int(n_experiment_ids * args.job_start_fraction): int(n_experiment_ids * args.job_end_fraction)]:

        # calculate resource needs based on ROI count
        unit_count = experiments_table.query('ecephys_session_id == @experiment_id').iloc[0]['unit_count']

        if args.force_overwrite or not already_fit(experiment_id, args.version):
            job_count += 1
            print('starting cluster job for {}, job count = {}'.format(experiment_id, job_count))
            job_title = 'oeid_{}_fit_glm_v_{}'.format(experiment_id, args.version)
            walltime = '{}:00:00'.format(int(np.ceil((calculate_required_walltime(unit_count)))))
            mem = '{}gb'.format(int(np.ceil((calculate_required_mem(unit_count)))))
            job_id = Slurm.JOB_ARRAY_ID
            job_array_id = Slurm.JOB_ARRAY_MASTER_ID
            output = stdout_location+"/"+str(job_array_id)+"_"+str(job_id)+"_"+str(experiment_id)+".out"
    
            # instantiate a SLURM object
            slurm = Slurm(
                cpus_per_task=16,
                job_name=job_title,
                time=walltime,
                mem=mem,
                output= output,
                partition="braintv"
            )

            args_string = job_string.format(experiment_id, args.version)
            slurm.sbatch('{} {} {}'.format(
                    python_executable,
                    python_file,
                    args_string,
                )
            )
            time.sleep(0.001)
