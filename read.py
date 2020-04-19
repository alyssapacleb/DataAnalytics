import pandas as pd

def read_data(keep=[], seed=1234567):
    """
    Helper method that loads data and does all pre-processing before feature engineering/analysis; 
    replicates the data munging done in `Actual_final_project.ipynb` in this repository.
    :params keep Keep these columns.
    :type list of strings
    
    :params seed Ensures replicability in the downsampling
    :type int or numpy.RandomState
    
    :return The DataFrame consisting of the 'working' copy of the data.
    :type pandas.DataFrame
    """
    url ='https://media.githubusercontent.com/media/fivethirtyeight/data/master/science-giving/science_federal_giving.csv'
    df = pd.read_csv(url, error_bad_lines=False)
    
    cols_to_drop = set(['cand_pty_affiliation', 'state','cand_office','cand_name', 'cleanedoccupation','cand_office_st',
                        'cleaned_name','employer','cmte_nm', 'cmte_id' , 'cmte_tp', 'cand_office_district',
                        'transaction_amt', 'cand_status', 'rpt_tp', 'transaction_tp', 'tran_id', 'file_num', 'memo_cd',
                        'memo_text', 'sub_id', 'other_id', 'zip_code', "cand_name", 'entity_tp',
                        'city','transaction_dt'])
    for col in keep:
        cols_to_drop.remove(col)
    
    # We don't care about these features because they don't give us additional information and they are redundant.
    new = df.drop(columns=list(cols_to_drop))
    
    # Standardize the label of the parties in the dataset
    new['cmte_pty'].replace(to_replace = 'dem', value = 'DEM', inplace = True)
    new['cmte_pty'].replace(to_replace = 'Dem', value = 'DEM', inplace = True)
    new['cmte_pty'].replace(to_replace = 'rep', value = 'REP', inplace = True)
    new['cmte_pty'].replace(to_replace = 'Rep', value = 'REP', inplace = True)

    # We only want to look at the general presidential election donations
    new = new[(new['cmte_pty'] == 'DEM') | (new['cmte_pty'] == 'REP')]

    # Keep only General and Primary election values in the transaction_pgi column
    new = new[(new['transaction_pgi'] == 'G') | (new['transaction_pgi'] == 'P')]

    new = new.dropna()
    new['cycle'] = new['cycle'].astype(int)
    
    # downsample
    only_dems = new[new['cmte_pty'] == 'DEM']
    only_reps = new[new['cmte_pty'] == 'REP']
    dem_part = only_dems.sample(frac = 0.05, random_state=seed) 
    rep_part = only_reps.sample(frac = 0.05, random_state=seed) 
    frames = [dem_part, rep_part] 
    stratified_sample = pd.concat(frames) 

    # df['2016_dollars'] = pd.to_numeric(df['2016_dollars'])
    # impute missing donations with mean
    # donation_mns = df['2016_dollars'].mean()
    # df['2016_dollars'].fillna(donation_mns, inplace = True)
    
    return stratified_sample

if __name__ == '__main__':
    print('Warning: import this module instead.')