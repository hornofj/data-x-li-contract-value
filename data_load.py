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


#load ir 
df_ir = pd.read_csv('../data-x-li-data/nss.csv')
# set #sc number as index 
df_ir  = df_ir.set_index('scnum')
df_ir.head()
df_ir.drop('Unnamed: 0', axis = 1, inplace = True)


# load all files
df1 = pd.read_csv('../data-x-li-data/scen_0001-0200.csv')
df2 = pd.read_csv('../data-x-li-data/scen_0201-0500.csv')
df3 = pd.read_csv('../data-x-li-data/scen_0501-0700.csv')
df4 = pd.read_csv('../data-x-li-data/scen_0701-1000.csv')
df5 = pd.read_csv('../data-x-li-data/scen_1001-1300.csv')
df6 = pd.read_csv('../data-x-li-data/scen_1301-1600.csv')
df7 = pd.read_csv('../data-x-li-data/scen_1601-1900.csv')
df8 = pd.read_csv('../data-x-li-data/scen_1901-2236.csv')

#making sample from them 
df_final = pd.DataFrame()
for i in range(1,8):
    #k = pd.DataFrame()
    df_final  = pd.concat([df_final, eval('df'+str(i)).sample(frac=0.2, replace = False, random_state = np.random.RandomState())])
# final df has 19 721 038 rows 
df_final.head()
# changing columns from cap to lower
df_final.columns = df_final.columns.str.lower()
# joined with ir scenarios 
df_merged = df_final.merge(df_ir, how = 'left',left_on = 'ir_scen', right_on = 'scnum')
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

# count month between two dates
def diff_month(d1, d2):
    return (d1.year - d2.year) * 12 + d1.month - d2.month

# delaring variable current date
cd = datetime.datetime(2020, 12, 31)
df_merged.head()
df_merged['inc_date_ct'] = df_merged['inc_date'].apply(lambda x: datetime.datetime.strptime(x, '%d/%m/%Y'))
df_merged['cnt_months'] = df_merged['inc_date_ct'].apply(lambda x: diff_month(cd, x))

df_merged.head()

# drop all columns thay are not used
df_merged = df_merged.drop(['inc_date', 'inc_date_ct', 'cv_ps_0'], axis = 1) 
df_merged.head()

SEED = 500
X_train, X_test, y_train, y_test= train_test_split(
    df_merged, #explanatory
    target, #response
    test_size=0.2, #hold out size
    random_state=SEED
    )

y_train.dtypes
#save testing and training data for later use (use list as a container)
with open(r"../data-x-li-data/df_merged_train_test.pickle", "wb") as output_file:
    pickle.dump([X_train, y_train, X_test, y_test], output_file)
