import pandas as pd 
import numpy as np
from statistics import mean
import math
from datetime import datetime
import statistics as st
import datetime
import sklearn 
from sklearn.model_selection import train_test_split

import pickle #load data in binary format


# load all files
df1 = pd.read_csv('../data-x-li-data/scen_0001-0200.csv')
df2 = pd.read_csv('../data-x-li-data/scen_0201-0500.csv')
df3 = pd.read_csv('../data-x-li-data/scen_0501-0700.csv')
df4 = pd.read_csv('../data-x-li-data/scen_0701-1000.csv')
df5 = pd.read_csv('../data-x-li-data/scen_1001-1300.csv')
df6 = pd.read_csv('../data-x-li-data/scen_1301-1600.csv')
df7 = pd.read_csv('../data-x-li-data/scen_1601-1900.csv')
df8 = pd.read_csv('../data-x-li-data/scen_1901-2236.csv')
df9 = pd.read_csv('../data-x-li-data/scen_3001-3300.csv')
df10 = pd.read_csv('../data-x-li-data/scen_3301-3600.csv')
df11 = pd.read_csv('../data-x-li-data/scen_3601-3900.csv')
df12 = pd.read_csv('../data-x-li-data/scen_3901-4200.csv')
df13 = pd.read_csv('../data-x-li-data/scen_4201-4700.csv')
df14 = pd.read_csv('../data-x-li-data/scen_4701-5000.csv')
df15 = pd.read_csv('../data-x-li-data/scen_5001-5236.csv')
df16 = pd.read_csv('../data-x-li-data/scen_6001-6300.csv')
df17 = pd.read_csv('../data-x-li-data/scen_6301-6600.csv')
df18 = pd.read_csv('../data-x-li-data/scen_6601-6900.csv')
df19 = pd.read_csv('../data-x-li-data/scen_6901-7200.csv')
df20 = pd.read_csv('../data-x-li-data/scen_7201-7500.csv')
df21 = pd.read_csv('../data-x-li-data/scen_7501-7800.csv')
df22 = pd.read_csv('../data-x-li-data/scen_7801-8236.csv')
df23 = pd.read_csv('../data-x-li-data/scen_9001-9300.csv')
df24 = pd.read_csv('../data-x-li-data/scen_9301-9700.csv')
df25 = pd.read_csv('../data-x-li-data/scen_9701-10000.csv')
df26 = pd.read_csv('../data-x-li-data/scen_10001-10300.csv')
df27 = pd.read_csv('../data-x-li-data/scen_10301-10600.csv')
df28 = pd.read_csv('../data-x-li-data/scen_10601-10900.csv')
df29 = pd.read_csv('../data-x-li-data/scen_10901-11236.csv')
df30 = pd.read_csv('../data-x-li-data/scen_12001-12300.csv')
df31 = pd.read_csv('../data-x-li-data/scen_12301-12600.csv')
df32 = pd.read_csv('../data-x-li-data/scen_12601-12900.csv')
df33 = pd.read_csv('../data-x-li-data/scen_12901-13200.csv')
df34 = pd.read_csv('../data-x-li-data/scen_13201-13500.csv')
df35 = pd.read_csv('../data-x-li-data/scen_13501-13800.csv')
df36 = pd.read_csv('../data-x-li-data/scen_13801-14236.csv')

# making sample from data
def df_generate_sample(fr_size, f_name):
    df_final = pd.DataFrame()
    for i in range(1,37):
        df_final  = pd.concat([df_final, eval('df'+str(i)).sample(frac=fr_size, replace = False, random_state = np.random.RandomState())])
    print(f_name, len(df_final))
    df_final.to_csv(f_name)
    return df_final

df_final05 = df_generate_sample(0.05, '../data-x-li-data/scen_sample_05p.csv')
df_final10 = df_generate_sample(0.1, '../data-x-li-data/scen_sample_10p.csv')
df_final20 = df_generate_sample(0.2, '../data-x-li-data/scen_sample_20p.csv')


# load ir 
df_ir = pd.read_csv('../data-x-li-data/nss.csv')
# set #sc number as index 
df_ir  = df_ir.set_index('scnum')
df_ir.head()
df_ir.drop('Unnamed: 0', axis = 1, inplace = True)

# count month between two dates
def diff_month(d1, d2):
    return (d1.year - d2.year) * 12 + d1.month - d2.month

def df_merge_nss(df, f_name):
    # changing columns from cap to lower
    df.columns = df.columns.str.lower()

    # joined with ir scenarios 
    df_merged = df.merge(df_ir, how = 'left', left_on = 'ir_scen', right_on = 'scnum')
    # drop column with the scenario number, is not relevant anymore
    df_merged.drop(['ir_scen', 'pol_num'], axis = 1, inplace = True)
    target = df_merged['pv_cf_rdr'].copy()

    # drop target from total df
    df_merged = df_merged.drop(['pv_cf_rdr'], axis = 1)
    df_merged.head()
    # std variable cv_ps_0_std
    mm = mean(df_merged['cv_ps_0'])
    std  = math.sqrt(st.variance(df_merged['cv_ps_0']))
    df_merged['cv_ps_0_std'] = df_merged['cv_ps_0'].apply(lambda x: (x-mm/std))
    df_merged.head()

    # declaring variable current date
    cd = datetime.datetime(2020, 12, 31)
    df_merged.head()
    df_merged['inc_date_ct'] = df_merged['inc_date'].apply(lambda x: datetime.datetime.strptime(x, '%d/%m/%Y'))
    df_merged['cnt_months'] = df_merged['inc_date_ct'].apply(lambda x: diff_month(cd, x))

    df_merged.head()

    # drop all columns thay are not used
    df_merged = df_merged.drop(['inc_date', 'inc_date_ct', 'cv_ps_0'], axis = 1) 
    df_merged.head()

    print(f_name, len(df_merged))
    df_merged.to_csv(f_name)
    return df_merged, target

df_merged05, target05 = df_merge_nss(df_final05, '../data-x-li-data/scen_sample_05p_nss.csv')
df_merged10, target10 = df_merge_nss(df_final10, '../data-x-li-data/scen_sample_10p_nss.csv')
df_merged20, target20 = df_merge_nss(df_final20, '../data-x-li-data/scen_sample_20p_nss.csv')


def df_split_to_pickle(df, target, f_name_s):
    SEED = 500
    X_train, X_test, y_train, y_test = train_test_split(
        df, #explanatory
        target, #response
        test_size=0.2, #hold out size
        random_state=SEED
        )

    y_train.dtypes
    #save testing and training data for later use (use list as a container)
    with open(r"../data-x-li-data/df_merged_train_test_"+f_name_s+".pickle", "wb") as output_file:
        pickle.dump([X_train, y_train, X_test, y_test], output_file)

df_split_to_pickle(df_merged05, target05, '05p')
df_split_to_pickle(df_merged10, target10, '10p')
df_split_to_pickle(df_merged20, target20, '20p')
