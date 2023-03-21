import numpy as np
import pandas as pd
import visual_behavior_glm.GLM_fit_tools as gft
import visual_behavior_glm.build_dataframes as bd
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from tqdm import tqdm

def dev():
    oeid = 956903412 

    run_params = {'include_invalid_rois':False}
    session = gft.load_data(oeid, run_params)

    cell_specimen_ids = session.cell_specimen_table.index.values   
    cell = d.get_cell_table(session, cell_specimen_ids[0])
    
    decode(cell)

def get_cells(session, data='events',window=[0,.75]):
    '''
        Iterate over all cells in this experiment and make a list
        of cell dataframes
    '''   
 
    # Iterate over all cells in experiment
    cells = []
    cell_specimen_ids = session.cell_specimen_table.index.values
    for cell in tqdm(cell_specimen_ids):
        # Generate the dataframe for this cell
        cell_df = get_cell_table(session, cell,data=data,window=window)
        cells.append(cell_df)
    
    # return list of cell dataframes
    return cells    
         

def get_cell_table(session, cell_specimen_id, data='events',window=[0,.75]):
    '''
        Generates a dataframe for one cell where rows are image presentations
        and the response at each timepoint is a column
    '''
    # Get cell activity interpolated onto constant time points
    df = bd.get_cell_df(session, cell_specimen_id, data=data)
    full_df = bd.get_cell_etr(df, session, time = window)
    
    # Pivot the table such that all responses at the same time point are a column
    cell = pd.pivot_table(full_df, values='response',
        index='stimulus_presentations_id', columns=['time'])
    
    # Annotate stimulus information
    cell = pd.merge(cell, session.stimulus_presentations, 
        left_index=True, right_index=True)

    return cell

def get_matrix(cell):
    '''
        Grabs the responses of the cell at each time point across each image presentation
        and returns a numpy matrix 
    '''
    cols = np.sort([x for x in cell.columns.values if not isinstance(x,str)])
    x = cell[cols].to_numpy()
    return x

def iterate_n_cells(cells):
    n_cells = [1,2,5,10,20,40,80,160,320]
   
    results = {} 
    for n in n_cells:
        if len(cells) < n:
            break
        print('Decoding with n={} cells'.format(n))
        results[n] = decode_cells(cells, n)

    results_df = pd.concat(results)
    return results_df

def decode_cells(cells, n_cells):
    '''
        Cells is a list of dataframes, one for each cell
        n_cells is the number of cells to decode with in each sample
    '''
    
    # How many times we need to sample in order to get 99% chance each cell 
    # is used at least once
    n_samples = int(np.ceil(np.log(0.01)/np.log(1-n_cells/len(cells))))
    
    # Iterate over samples and save output
    output = []
    for n in tqdm(range(0,n_samples)):
        temp = decode_cells_sample(cells, n_cells)
        output.append(temp)

    # Return the output of the samples
    output_df = pd.DataFrame(output)
    output_df['n_cells'] = n_cells
    return output_df


def decode_cells_sample(cells, n_cells):
    '''
        Sample from the list of cells, and perform decoding once
        returns the output of the decoding
    '''
 
    # Sample n_cells from list of cells
    cells_in_sample = np.random.choice(len(cells),n_cells, replace=False)
    sample_cells = [cells[i] for i in cells_in_sample] 
    
    # Construct X 
    X = []
    for cell in sample_cells:
        X.append(get_matrix(cell))
    X = np.concatenate(X,axis=1)
    
    # y is the same for every cell, since its a behavioral output
    y = sample_cells[0]['is_change'].values

    # run CV model
    clf = RandomForestClassifier(class_weight='balanced')
    clf.fit(X,y)
    model = {}
    model['score'] = clf.score(X,y)
    model['prediction'] = clf.predict(X)   
    model['behavior_correlation'] = np.corrcoef(y,model['prediction'])[1,0] 
    model['decoder'] = clf
    
    # return decoder
    return model 
    


