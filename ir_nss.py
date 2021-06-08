import pandas as pd 
import numpy as np
from nelson_siegel_svensson.calibrate import calibrate_ns_ols
import matplotlib.pyplot as plt
    
# pip install nelson_siegel_svensson


df_all = pd.read_csv('inputs_ir.csv')  
df_all.head()
ir_sc = list(df_all["SC_NUM"].unique())
# this doesn't change
ir_year = np.array(range(1,51))

d = {'scnum': ir_sc, 
        'beta0': np.nan, 
        'beta1': np.nan,
        'beta2': np.nan, 
        'tau': np.nan}

df_all['DISC_R'] = df_all['DISC_R'].apply(lambda x: round(x, 5))

df_ns = pd.DataFrame(d)
#take 1 scenario
for i in range(len(ir_sc)):
    # take all the ir where the scenaio is i and make an array 
    ir_temp = np.array(df_all['DISC_R'][df_all["SC_NUM"]==ir_sc[i]])
    # this magic function
    curve, status = calibrate_ns_ols(ir_year, ir_temp, tau0=2.0)
    # put the output in the columns
    df_ns.loc[i,'beta0'] = curve.beta0
    df_ns.loc[i,'beta1'] = curve.beta1
    df_ns.loc[i, 'beta2'] = curve.beta2
    df_ns.loc[i, 'tau'] = curve.tau

print(df_ns)

df_ns.to_csv('../data-x-li-data/nss.csv')