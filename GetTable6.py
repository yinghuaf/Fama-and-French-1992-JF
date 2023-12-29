import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from pandas.tseries.offsets import *
import statsmodels.api as sm

df = pd.read_csv('df4.csv',index_col=0)
df['datadate'] = pd.to_datetime(df['datadate'])
df['const'] = 1

table = pd.DataFrame()

# market returns
df_temp = df[df['exchg']==11]

port = pd.DataFrame(df_temp.groupby('datadate').apply(lambda x: np.sum(x['ME']/np.sum(x['ME'])*x['trt1m'])) )
table.at['NYSE-VW','All_mean'] = np.mean(port)[0]
table.at['NYSE-VW','All_std'] = np.std(port)[0]
table.at['NYSE-VW','All_t'] = np.mean(port)[0]/np.std(port)[0]*np.sqrt(len(port))

port = pd.DataFrame(df_temp.groupby('datadate')['trt1m'].mean())
table.at['NYSE-EW','All_mean'] = np.mean(port)[0]
table.at['NYSE-EW','All_std'] = np.std(port)[0]
table.at['NYSE-EW','All_t'] = np.mean(port)[0]/np.std(port)[0]*np.sqrt(len(port))

df_temp = df[(df['exchg']==11) & (df['datadate']<=pd.to_datetime('1976-12-31'))]

port = pd.DataFrame(df_temp.groupby('datadate').apply(lambda x: np.sum(x['ME']/np.sum(x['ME'])*x['trt1m'])) )
table.at['NYSE-VW','p1_mean'] = np.mean(port)[0]
table.at['NYSE-VW','p1_std'] = np.std(port)[0]
table.at['NYSE-VW','p1_t'] = np.mean(port)[0]/np.std(port)[0]*np.sqrt(len(port))

port = pd.DataFrame(df_temp.groupby('datadate')['trt1m'].mean())
table.at['NYSE-EW','p1_mean'] = np.mean(port)[0]
table.at['NYSE-EW','p1_std'] = np.std(port)[0]
table.at['NYSE-EW','p1_t'] = np.mean(port)[0]/np.std(port)[0]*np.sqrt(len(port))

df_temp = df[(df['exchg']==11) & (df['datadate']>pd.to_datetime('1976-12-31'))]

port = pd.DataFrame(df_temp.groupby('datadate').apply(lambda x: np.sum(x['ME']/np.sum(x['ME'])*x['trt1m'])) )
table.at['NYSE-VW','p2_mean'] = np.mean(port)[0]
table.at['NYSE-VW','p2_std'] = np.std(port)[0]
table.at['NYSE-VW','p2_t'] = np.mean(port)[0]/np.std(port)[0]*np.sqrt(len(port))

port = pd.DataFrame(df_temp.groupby('datadate')['trt1m'].mean())
table.at['NYSE-EW','p2_mean'] = np.mean(port)[0]
table.at['NYSE-EW','p2_std'] = np.std(port)[0]
table.at['NYSE-EW','p2_t'] = np.mean(port)[0]/np.std(port)[0]*np.sqrt(len(port))

x = ['const','ln(ME)','ln(BE/ME)']
y = ['trt1m']

df_temp = df
df_temp = df_temp.dropna()
dates = pd.unique(df_temp['datadate'])

table_params = pd.DataFrame()
table_bse = pd.DataFrame()
table_t = pd.DataFrame()

for date in dates:
    df_temp_temp = df_temp[df_temp['datadate']==date]
    df_temp_temp = df_temp_temp[y+x]
    df_temp_temp = df_temp_temp.drop_duplicates()
    
    try:
        result = sm.OLS(df_temp_temp[y], df_temp_temp[x], missing='drop').fit()
    except:
        continue
    
    table_params = table_params.append(pd.DataFrame(result.params,columns=[date]).T)
    table_bse = table_bse.append(pd.DataFrame(result.bse, columns=[date]).T)
    table_t = table_t.append(pd.DataFrame(result.tvalues, columns=[date]).T)

table.at['model1-a','All_mean'] = table_params.mean()[0]
table.at['model1-b2','All_mean'] = table_params.mean()[1]
table.at['model1-b3','All_mean'] = table_params.mean()[2]

table.at['model1-a','All_std'] = table_bse.mean()[0]
table.at['model1-b2','All_std'] = table_bse.mean()[1]
table.at['model1-b3','All_std'] = table_bse.mean()[2]

table.at['model1-a','All_t'] = table_t.mean()[0]
table.at['model1-b2','All_t'] = table_t.mean()[1]
table.at['model1-b3','All_t'] = table_t.mean()[2]

##########################
df_temp = df[df['datadate']<=pd.to_datetime('1976-12-31')]
df_temp = df_temp.dropna()
dates = pd.unique(df_temp['datadate'])

table_params = pd.DataFrame()
table_bse = pd.DataFrame()
table_t = pd.DataFrame()

for date in dates:
    df_temp_temp = df_temp[df_temp['datadate']==date]
    df_temp_temp = df_temp_temp[y+x]
    df_temp_temp = df_temp_temp.drop_duplicates()
    
    try:
        result = sm.OLS(df_temp_temp[y], df_temp_temp[x], missing='drop').fit()
    except:
        continue
    
    table_params = table_params.append(pd.DataFrame(result.params,columns=[date]).T)
    table_bse = table_bse.append(pd.DataFrame(result.bse, columns=[date]).T)
    table_t = table_t.append(pd.DataFrame(result.tvalues, columns=[date]).T)

table.at['model1-a','p1_mean'] = table_params.mean()[0]
table.at['model1-b2','p1_mean'] = table_params.mean()[1]
table.at['model1-b3','p1_mean'] = table_params.mean()[2]

table.at['model1-a','p1_std'] = table_bse.mean()[0]
table.at['model1-b2','p1_std'] = table_bse.mean()[1]
table.at['model1-b3','p1_std'] = table_bse.mean()[2]

table.at['model1-a','p1_t'] = table_t.mean()[0]
table.at['model1-b2','p1_t'] = table_t.mean()[1]
table.at['model1-b3','p1_t'] = table_t.mean()[2]

##########################
df_temp = df[df['datadate']>pd.to_datetime('1976-12-31')]
df_temp = df_temp.dropna()
dates = pd.unique(df_temp['datadate'])

table_params = pd.DataFrame()
table_bse = pd.DataFrame()
table_t = pd.DataFrame()

for date in dates:
    df_temp_temp = df_temp[df_temp['datadate']==date]
    df_temp_temp = df_temp_temp[y+x]
    df_temp_temp = df_temp_temp.drop_duplicates()
    
    try:
        result = sm.OLS(df_temp_temp[y], df_temp_temp[x], missing='drop').fit()
    except:
        continue
    
    table_params = table_params.append(pd.DataFrame(result.params,columns=[date]).T)
    table_bse = table_bse.append(pd.DataFrame(result.bse, columns=[date]).T)
    table_t = table_t.append(pd.DataFrame(result.tvalues, columns=[date]).T)

table.at['model1-a','p2_mean'] = table_params.mean()[0]
table.at['model1-b2','p2_mean'] = table_params.mean()[1]
table.at['model1-b3','p2_mean'] = table_params.mean()[2]

table.at['model1-a','p2_std'] = table_bse.mean()[0]
table.at['model1-b2','p2_std'] = table_bse.mean()[1]
table.at['model1-b3','p2_std'] = table_bse.mean()[2]

table.at['model1-a','p2_t'] = table_t.mean()[0]
table.at['model1-b2','p2_t'] = table_t.mean()[1]
table.at['model1-b3','p2_t'] = table_t.mean()[2]

x = ['const','beta','ln(ME)','ln(BE/ME)']
y = ['trt1m']

df_temp = df
df_temp = df_temp.dropna()
dates = pd.unique(df_temp['datadate'])

table_params = pd.DataFrame()
table_bse = pd.DataFrame()
table_t = pd.DataFrame()

for date in dates:
    df_temp_temp = df_temp[df_temp['datadate']==date]
    df_temp_temp = df_temp_temp[y+x]
    df_temp_temp = df_temp_temp.drop_duplicates()
    
    try:
        result = sm.OLS(df_temp_temp[y], df_temp_temp[x], missing='drop').fit()
    except:
        continue
    
    table_params = table_params.append(pd.DataFrame(result.params,columns=[date]).T)
    table_bse = table_bse.append(pd.DataFrame(result.bse, columns=[date]).T)
    table_t = table_t.append(pd.DataFrame(result.tvalues, columns=[date]).T)

table.at['model2-a','All_mean'] = table_params.mean()[0]
table.at['model2-b1','All_mean'] = table_params.mean()[1]
table.at['model2-b2','All_mean'] = table_params.mean()[2]
table.at['model2-b3','All_mean'] = table_params.mean()[3]

table.at['model2-a','All_std'] = table_bse.mean()[0]
table.at['model2-b1','All_std'] = table_bse.mean()[1]
table.at['model2-b2','All_std'] = table_bse.mean()[2]
table.at['model2-b3','All_std'] = table_bse.mean()[3]

table.at['model2-a','All_t'] = table_t.mean()[0]
table.at['model2-b1','All_t'] = table_t.mean()[1]
table.at['model2-b2','All_t'] = table_t.mean()[2]
table.at['model2-b3','All_t'] = table_t.mean()[3]

##########################
df_temp = df[df['datadate']<=pd.to_datetime('1976-12-31')]
df_temp = df_temp.dropna()
dates = pd.unique(df_temp['datadate'])

table_params = pd.DataFrame()
table_bse = pd.DataFrame()
table_t = pd.DataFrame()

for date in dates:
    df_temp_temp = df_temp[df_temp['datadate']==date]
    df_temp_temp = df_temp_temp[y+x]
    df_temp_temp = df_temp_temp.drop_duplicates()
    
    try:
        result = sm.OLS(df_temp_temp[y], df_temp_temp[x], missing='drop').fit()
    except:
        continue
    
    table_params = table_params.append(pd.DataFrame(result.params,columns=[date]).T)
    table_bse = table_bse.append(pd.DataFrame(result.bse, columns=[date]).T)
    table_t = table_t.append(pd.DataFrame(result.tvalues, columns=[date]).T)

table.at['model2-a','p1_mean'] = table_params.mean()[0]
table.at['model2-b1','p1_mean'] = table_params.mean()[1]
table.at['model2-b2','p1_mean'] = table_params.mean()[2]
table.at['model2-b3','p1_mean'] = table_params.mean()[3]

table.at['model2-a','p1_std'] = table_bse.mean()[0]
table.at['model2-b1','p1_std'] = table_bse.mean()[1]
table.at['model2-b2','p1_std'] = table_bse.mean()[2]
table.at['model2-b3','p1_std'] = table_bse.mean()[3]

table.at['model2-a','p1_t'] = table_t.mean()[0]
table.at['model2-b1','p1_t'] = table_t.mean()[1]
table.at['model2-b2','p1_t'] = table_t.mean()[2]
table.at['model2-b3','p1_t'] = table_t.mean()[3]

##########################
df_temp = df[df['datadate']>pd.to_datetime('1976-12-31')]
df_temp = df_temp.dropna()
dates = pd.unique(df_temp['datadate'])

table_params = pd.DataFrame()
table_bse = pd.DataFrame()
table_t = pd.DataFrame()

for date in dates:
    df_temp_temp = df_temp[df_temp['datadate']==date]
    df_temp_temp = df_temp_temp[y+x]
    df_temp_temp = df_temp_temp.drop_duplicates()
    
    try:
        result = sm.OLS(df_temp_temp[y], df_temp_temp[x], missing='drop').fit()
    except:
        continue
    
    table_params = table_params.append(pd.DataFrame(result.params,columns=[date]).T)
    table_bse = table_bse.append(pd.DataFrame(result.bse, columns=[date]).T)
    table_t = table_t.append(pd.DataFrame(result.tvalues, columns=[date]).T)

table.at['model2-a','p2_mean'] = table_params.mean()[0]
table.at['model2-b1','p2_mean'] = table_params.mean()[1]
table.at['model2-b2','p2_mean'] = table_params.mean()[2]
table.at['model2-b3','p2_mean'] = table_params.mean()[3]

table.at['model2-a','p2_std'] = table_bse.mean()[0]
table.at['model2-b1','p2_std'] = table_bse.mean()[1]
table.at['model2-b2','p2_std'] = table_bse.mean()[2]
table.at['model2-b3','p2_std'] = table_bse.mean()[3]

table.at['model2-a','p2_t'] = table_t.mean()[0]
table.at['model2-b1','p2_t'] = table_t.mean()[1]
table.at['model2-b2','p2_t'] = table_t.mean()[2]
table.at['model2-b3','p2_t'] = table_t.mean()[3]

table = round(table,2)

table.to_csv('Table6.csv')