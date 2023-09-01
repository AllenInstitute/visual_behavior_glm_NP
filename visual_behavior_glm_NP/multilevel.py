import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import statsmodels.formula.api as smf
import pickle

PATH = '/home/alex.piet/codebase/behavior/PSTH/'

def multilevel_dev():
    # Generate data
    #y = b*x + m 
    b = 2
    n = 100
    obs = []
    for i in range(0,n):
        m = np.random.randn()+ 2*np.mod(i,2)
        xs = np.random.rand(10)*5
        for x in xs:
            obs.append({'i':i,'x':x,'y':b*x+m+np.random.randn()*.01,'group':np.mod(i,2)})
    obs = pd.DataFrame(obs)

    # Plot the raw data    
    fig, ax = plt.subplots(2,3)
    ax[0,0].plot(obs['x'],obs['y'],'ko',alpha=.5)
    ax[0,1].plot(obs.query('group == 0')['x'],obs.query('group == 0')['y'],'ro',alpha=.5)
    ax[0,1].plot(obs.query('group == 1')['x'],obs.query('group == 1')['y'],'bo',alpha=.5)
    for i in range(0,n):
        ax[0,2].plot(obs.query('i==@i')['x'],obs.query('i==@i')['y'],'o',alpha=.5)

    # do different types of regression
    regression(obs,ax[1,0],'k')
    regression(obs.query('group == 0'), ax[1,1],'r')
    regression(obs.query('group == 1'), ax[1,1],'b')
    fits = []
    for i in range(0,n):
        model = regression(obs.query('i==@i'),ax[1,2],np.random.rand(3,))
        fits.append({'i':i,'intercept':model.intercept_})
    fits = pd.DataFrame(fits)
    
    # Do mixed linear regression
    model = smf.mixedlm("y ~ x",
        vc_formula={'i':'0+C(i)'},
        re_formula='1',
        groups='group',
        data=obs).fit()
    print(model.summary()) 
    #bic = k*ln(n)-2*ln(L)
    bic1 = 4*np.log(len(obs))-2*model.llf
    print(bic1)

    model = smf.mixedlm("y ~ x",
        re_formula='1',
        groups='i',
        data=obs).fit()
    print(model.summary()) 
    bic2 = 3*np.log(len(obs))-2*model.llf
    print(bic2)
    print('Delta BIC: {}'.format(bic1-bic2))

    plt.tight_layout()    
    return obs,pd.DataFrame(fits)

def regression(obs,ax,color):
    x = obs['x'].values.reshape((-1,1))
    y = obs['y'].values
    model = LinearRegression(fit_intercept=True)
    model.fit(x,y)
    ax.plot(0,model.coef_,'o',color=color,alpha=.5)
    ax.plot(1,model.intercept_,'o',color=color,alpha=.5)
    ax.set_xlim(-.5,1.5)
    ax.set_ylim(-1,3)
    ax.set_xticks([0,1])
    ax.set_xticklabels(['slope','intercept'],rotation=60)
    
    return model

def run_vip_miss(testing=False):
    print('Loading vip data')
    df = pd.read_feather('/allen/programs/braintv/workgroups/nc-ophys/visual_behavior/ophys_glm/image_dfs/events/summary_Vip-IRES-Cre_second_half.feather')
    
    print('Loading summary data')
    summary_df = pd.read_pickle('/allen/programs/braintv/workgroups/nc-ophys/alex.piet/behavior/psy_fits_v21/summary_data/_summary_table.pkl')

    print('Filtering, merging vip data')
    cols = ['behavior_session_id','visual_strategy_session','experience_level','equipment_name']
    df = pd.merge(df, summary_df[cols],on='behavior_session_id') 
    df = df.query('(pre_miss_1 ==1)').copy()
    familiar_summary_df = summary_df.query('experience_level == "Familiar"')
    familiar_bsid = familiar_summary_df['behavior_session_id'].unique()
    df.drop(df[~df['behavior_session_id'].isin(familiar_bsid)].index, inplace=True)
    df = df.query('equipment_name == "MESO.1"').copy()

    print('Filtering specific running speeds')
    df = df.query('running_speed > 20').query('running_speed < 50').copy()
    df['strategy'] = ['visual' if x else 'timing' for x in df['visual_strategy_session']]
    df = df[['strategy','ophys_experiment_id','cell_specimen_id',
        'running_speed','response']]
    
    if testing: 
        print('Sampling')
        df = df.sample(frac=.1)

    print('Fitting strategy model')
    strategy_model = fit_strategy_model(df)
    filename = PATH+'vip_miss_strategy_model.pkl'
    with open(filename, 'wb') as file:
        pickle.dump(strategy_model, file)
        print(f'Model saved to "{filename}"')

    print('Fitting cell model')
    cell_model = fit_cell_model(df)
    filename = PATH+'vip_miss_cell_model.pkl'
    with open(filename, 'wb') as file:
        pickle.dump(cell_model, file)
        print(f'Model saved to "{filename}"')
    
    delta_bic = strategy_model.bic - cell_model.bic
    print('Delta BIC: {}'.format(delta_bic))

    return df, strategy_model, cell_model



def run_vip_hit(testing=False):
    print('Loading vip data')
    df = pd.read_feather('/allen/programs/braintv/workgroups/nc-ophys/visual_behavior/ophys_glm/image_dfs/events/summary_Vip-IRES-Cre_second_half.feather')
    
    print('Loading summary data')
    summary_df = pd.read_pickle('/allen/programs/braintv/workgroups/nc-ophys/alex.piet/behavior/psy_fits_v21/summary_data/_summary_table.pkl')

    print('Filtering, merging vip data')
    cols = ['behavior_session_id','visual_strategy_session','experience_level','equipment_name']
    df = pd.merge(df, summary_df[cols],on='behavior_session_id') 
    df = df.query('(pre_hit_1 ==1)').copy()
    familiar_summary_df = summary_df.query('experience_level == "Familiar"')
    familiar_bsid = familiar_summary_df['behavior_session_id'].unique()
    df.drop(df[~df['behavior_session_id'].isin(familiar_bsid)].index, inplace=True)
    df = df.query('equipment_name == "MESO.1"').copy()

    print('Filtering specific running speeds')
    df = df.query('running_speed > 10').query('running_speed < 50').copy()
    df['strategy'] = ['visual' if x else 'timing' for x in df['visual_strategy_session']]
    df = df[['strategy','ophys_experiment_id','cell_specimen_id',
        'running_speed','response']]
    
    if testing: 
        print('Sampling')
        df = df.sample(frac=.1)

    print('Fitting strategy model')
    strategy_model = fit_strategy_model(df)
    filename = PATH+'vip_hit_strategy_model.pkl'
    with open(filename, 'wb') as file:
        pickle.dump(strategy_model, file)
        print(f'Model saved to "{filename}"')

    print('Fitting cell model')
    cell_model = fit_cell_model(df)
    filename = PATH+'vip_hit_cell_model.pkl'
    with open(filename, 'wb') as file:
        pickle.dump(cell_model, file)
        print(f'Model saved to "{filename}"')
    
    delta_bic = strategy_model.bic - cell_model.bic
    print('Delta BIC: {}'.format(delta_bic))

    return df, strategy_model, cell_model

def run_vip_omission(testing=False):
    print('Loading vip data')
    df = pd.read_feather('/allen/programs/braintv/workgroups/nc-ophys/visual_behavior/ophys_glm/image_dfs/events/summary_Vip-IRES-Cre_second_half.feather')
    
    print('Loading summary data')
    summary_df = pd.read_pickle('/allen/programs/braintv/workgroups/nc-ophys/alex.piet/behavior/psy_fits_v21/summary_data/_summary_table.pkl')

    print('Filtering, merging vip data')
    cols = ['behavior_session_id','visual_strategy_session','experience_level','equipment_name']
    df = pd.merge(df, summary_df[cols],on='behavior_session_id') 
    df.drop(df[~df['omitted']].index,inplace=True)
    familiar_summary_df = summary_df.query('experience_level == "Familiar"')
    familiar_bsid = familiar_summary_df['behavior_session_id'].unique()
    df.drop(df[~df['behavior_session_id'].isin(familiar_bsid)].index, inplace=True)
    df = df.query('equipment_name == "MESO.1"').copy()

    print('Filtering specific running speeds')
    df = df.query('running_speed > 10').query('running_speed < 50').copy()
    df['strategy'] = ['visual' if x else 'timing' for x in df['visual_strategy_session']]
    df = df[['strategy','ophys_experiment_id','cell_specimen_id',
        'running_speed','response']]
    
    if testing: 
        print('Sampling')
        df = df.sample(frac=.001)

    print('Fitting strategy model')
    strategy_model = fit_strategy_model(df)
    filename = PATH+'vip_omission_strategy_model.pkl'
    with open(filename, 'wb') as file:
        pickle.dump(strategy_model, file)
        print(f'Model saved to "{filename}"')

    print('Fitting cell model')
    cell_model = fit_cell_model(df)
    filename = PATH+'vip_omission_cell_model.pkl'
    with open(filename, 'wb') as file:
        pickle.dump(cell_model, file)
        print(f'Model saved to "{filename}"')
    
    delta_bic = strategy_model.bic - cell_model.bic
    print('Delta BIC: {}'.format(delta_bic))

    return df, strategy_model, cell_model



def run_vip_image(testing=False):
    print('Loading vip data')
    df = pd.read_feather('/allen/programs/braintv/workgroups/nc-ophys/visual_behavior/ophys_glm/image_dfs/events/summary_Vip-IRES-Cre_second_half.feather')
    
    print('Loading summary data')
    summary_df = pd.read_pickle('/allen/programs/braintv/workgroups/nc-ophys/alex.piet/behavior/psy_fits_v21/summary_data/_summary_table.pkl')

    print('Filtering, merging vip data')
    cols = ['behavior_session_id','visual_strategy_session','experience_level','equipment_name']
    df = pd.merge(df, summary_df[cols],on='behavior_session_id') 
    df.drop(df[df['is_change'] | df['omitted']].index,inplace=True)
    familiar_summary_df = summary_df.query('experience_level == "Familiar"')
    familiar_bsid = familiar_summary_df['behavior_session_id'].unique()
    df.drop(df[~df['behavior_session_id'].isin(familiar_bsid)].index, inplace=True)
    df = df.query('equipment_name == "MESO.1"').copy()

    print('Filtering specific funning speeds')
    df = df.query('running_speed > 30').query('running_speed < 50').copy()
    df['strategy'] = ['visual' if x else 'timing' for x in df['visual_strategy_session']]
    df = df[['strategy','ophys_experiment_id','cell_specimen_id',
        'running_speed','response']]
   
    if testing: 
        print('Sampling')
        df = df.sample(frac=.75)

    print('Fitting strategy model')
    strategy_model = fit_strategy_model(df)
    filename = PATH+'vip_image_strategy_model.pkl'
    with open(filename, 'wb') as file:
        pickle.dump(strategy_model, file)
        print(f'Model saved to "{filename}"')

    print('Fitting cell model')
    cell_model = fit_cell_model(df)
    filename = PATH+'vip_image_cell_model.pkl'
    with open(filename, 'wb') as file:
        pickle.dump(cell_model, file)
        print(f'Model saved to "{filename}"')
    
    delta_bic = strategy_model.bic - cell_model.bic
    print('Delta BIC: {}'.format(delta_bic))

    return df, strategy_model, cell_model


def fit_strategy_model(df):
    model = smf.mixedlm("response ~ running_speed",
        vc_formula={'cell_specimen_id':'0+C(cell_specimen_id)'},
        re_formula='1',
        groups='strategy',
        data=df).fit()
    print(model.summary()) 
    model.bic = 4*np.log(len(df))-2*model.llf
    print('BIC: {}'.format(model.bic))
    return model

def fit_cell_model(df):
    model = smf.mixedlm("response ~ running_speed",
        re_formula='1',
        groups='cell_specimen_id',
        data=df).fit()
    print(model.summary()) 
    model.bic = 3*np.log(len(df))-2*model.llf
    print('BIC: {}'.format(model.bic))
    return model

def load_results(save=False):
    types = ['image','omission','hit','miss']
    results = []
    for t in types:
        cell_filename = PATH+'vip_{}_cell_model.pkl'.format(t)
        strategy_filename = PATH+'vip_{}_strategy_model.pkl'.format(t)
        
        with open(strategy_filename, 'rb') as f:
            strategy = pickle.load(f)

        with open(cell_filename, 'rb') as f:
            cell = pickle.load(f)       
        delta_bic = strategy.bic - cell.bic
        results.append({'type':t,'delta_bic':delta_bic})
    results = pd.DataFrame(results)
    if save:
        results.to_csv(PATH+'vip_summary_multilevel_regression.csv',index=False)


