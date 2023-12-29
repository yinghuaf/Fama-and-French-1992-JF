import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from pandas.tseries.offsets import *
import statsmodels.api as sm

def get_OLS_table(AH, X, y_name, date_name):
    df = pd.DataFrame()

    y = [y_name]

    for i in range(1,len(X)+1):
        x = X['x'+str(i)]
        results = AHP_panel(data=AH,x=x,y=y, date_name='Trddt')
        df_temp = save_OLS(results,'table_AH_panel_explain_level_p2')
    
        for var in df_temp.index:
            df.at[var,'('+str(i)+')'] =  str(df_temp.at[var,'parameter'])
            
            if (df_temp.at[var,'pvalue']<=0.1) & (df_temp.at[var,'pvalue']>0.05):
                df.at[var,'('+str(i)+')'] =  str(df_temp.at[var,'parameter'])+'*'
            if (df_temp.at[var,'pvalue']<=0.05) & (df_temp.at[var,'pvalue']>0.01):
                df.at[var,'('+str(i)+')'] =  str(df_temp.at[var,'parameter'])+'**'
            if df_temp.at[var,'pvalue']<=0.01:
                df.at[var,'('+str(i)+')'] =  str(df_temp.at[var,'parameter'])+'***'
                
            df.at['('+var+')','('+str(i)+')'] =  '('+ str(df_temp.at[var,'pvalue']) + ')_'
            
        df.at['r2','('+str(i)+')'] = df_temp['r2'][0]
                
    df = df.fillna('')
    
    # move the R2 row to the last row
    temp_r2 = pd.DataFrame(df.loc['r2']).T
    df = df.drop('r2')
    df = df.append(temp_r2 )
    
    return df

def save_OLS(results,save_name):
    df = pd.merge(round(pd.DataFrame(results.params),3),round(pd.DataFrame(results.tstats),2),
              how='left', left_index=True, right_index=True)

    df = pd.merge(df, round(pd.DataFrame(results.pvalues),3), how='left', left_index=True, right_index=True)
    df['r2'] = round(results.rsquared*100,2)

    return df

df = pd.read_csv('df2.csv',index_col=0)
df['datadate'] = pd.to_datetime(df['datadate'])

X = {}

X['x1'] = ['beta']
X['x2'] = ['ln(ME)']
X['x3'] = ['beta','ln(ME)']
X['x4'] = ['ln(BE/ME)']
X['x5'] = ['ln(A/ME)','ln(A/BE)']
X['x6'] = ['E/P dummy','E(+)/P']
X['x7'] = ['ln(ME)', 'ln(BE/ME)']
X['x8'] = ['ln(ME)', 'ln(A/ME)','ln(A/BE)']
X['x9'] = ['ln(ME)', 'E/P dummy','E(+)/P']
X['x10'] = ['ln(ME)', 'ln(BE/ME)', 'E/P dummy','E(+)/P']
X['x11'] = ['ln(ME)', 'ln(A/ME)','ln(A/BE)', 'E/P dummy','E(+)/P']

y_name = 'trt1m'
data = df
date_name = 'datadate'

table = pd.DataFrame()
y = [y_name]

dates = pd.unique(data[date_name])

for i in range(1,len(X)+1):
    x = X['x'+str(i)]
    df_params = pd.DataFrame()
    df_t = pd.DataFrame()
    
    for date in dates:
        data_temp = data[data[date_name]==date]
        data_temp = data_temp[y+x]
        data_temp = data_temp.drop_duplicates()
        
        if len(data_temp)<=7:
            continue
        try:
            result = sm.OLS(data_temp[y], sm.add_constant(data_temp[x]), missing='drop').fit()
        except:
            continue
            
        df_params = df_params.append(pd.DataFrame(result.params, columns=[date]).T)
        df_t = df_t.append(pd.DataFrame(result.params, columns=[date]).T/pd.DataFrame(result.bse, columns=[date]).T)
        
    df_params = np.mean(df_params)
    df_t = np.mean(df_t)
    
    for var in df_params.index:
        table.at[str(i), var] = round(df_params[var],2)
        table.at['('+str(i)+')', var] = '('+ str(round(df_t[var]*10,2)) + ')_'

table.to_csv('Table3.csv')