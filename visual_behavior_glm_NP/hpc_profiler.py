import os
import pandas as pd

def profile_jobs(version):
    # switch to directory
    # process log files to get slurm ids
    # for each id, use seff to get used efficiency
    # save everything to dataframe

    directory = '/home/alex.piet/codebase/NP/ephys/logs/job_records_{}'.format(version)
    files = os.listdir(directory)
    slurm_ids = [x.split('_')[0] for x in files]
    cpu = []
    mem = []
    mem_used = []
    for slurm_id in slurm_ids:
        output = os.popen('seff {}'.format(slurm_id)).read()
        cpu.append(output.split('CPU Efficiency: ')[1].split('%')[0])
        mem.append(output.split('Memory Efficiency: ')[1].split('%')[0])
        mem_used.append(output.split('Memory Utilized: ')[1].split(' GB')[0])
    results = pd.DataFrame({'cpu eff.':cpu,'mem eff. ':mem,'mem used':mem_used})
    return results
