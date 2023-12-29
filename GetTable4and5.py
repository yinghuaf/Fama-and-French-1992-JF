import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from pandas.tseries.offsets import *
import statsmodels.api as sm

def get_ffill(data, var_name, date_name, stock_name):
    data = data.sort_values(by=[stock_name, date_name])
    data = data.reset_index(drop=True)

    temp = pd.DataFrame(data.groupby(stock_name)[var_name].fillna(method='ffill'))

    del data[var_name]

    data = pd.merge(data, temp, how='left', left_index=True, right_index=True)
    return data

def get_uni_port_value(data, rank_name, var_name, date_name, rename):
    data[var_name] = data[var_name].fillna(0)
    temp = data.groupby([date_name, rank_name])[var_name].mean().reset_index(drop=False)
    temp = pd.DataFrame(temp.groupby(rank_name)[var_name].mean()).rename(columns={var_name:rename}).T
    return temp

def get_var_rank(data, exchange_name, exchang_code, var_name, date_name, stock_name):
    data_temp = data[(data[exchange_name]==exchang_code) & (data[date_name].astype(str).str[-5:]=='06-30')]

# create quantile table
    for quantile in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        if quantile==0.1:
            quantile_table = pd.DataFrame(data_temp.groupby(date_name)[var_name].quantile(quantile)).rename(columns={var_name:str('q'+str(int(quantile*10)))})
        else: 
            quantile_table = pd.merge(quantile_table, pd.DataFrame(data_temp.groupby(date_name)[var_name].quantile(quantile)).rename(columns={var_name:str('q'+str(int(quantile*10)))}),
                                 how='left', left_index=True, right_index=True)

    data[date_name+'_june'] = pd.to_datetime(data[date_name].astype(str).str[:-5] + '06-30')
    data.loc[data[date_name]>data[date_name+'_june'], date_name+'_june'] = pd.to_datetime((data[data[date_name]>data[date_name+'_june']][date_name+'_june'].astype(str).str[:-6].astype(int)+1).astype(str) + '-06-30')

    data = pd.merge(data, quantile_table, how='left', left_on='datadate_june', right_index=True)

    data[var_name + '_rank'] = 0
    data.loc[data[var_name]<=data['q1'] , var_name + '_rank']= 1
    data.loc[(data[var_name]>data['q1']) & (data[var_name]<=data['q2']) , var_name + '_rank']= 2
    data.loc[(data[var_name]>data['q2']) & (data[var_name]<=data['q3']) , var_name + '_rank']= 3
    data.loc[(data[var_name]>data['q3']) & (data[var_name]<=data['q4']) , var_name + '_rank']= 4
    data.loc[(data[var_name]>data['q4']) & (data[var_name]<=data['q5']) , var_name + '_rank']= 5
    data.loc[(data[var_name]>data['q5']) & (data[var_name]<=data['q6']) , var_name + '_rank']= 6
    data.loc[(data[var_name]>data['q6']) & (data[var_name]<=data['q7']) , var_name + '_rank']= 7
    data.loc[(data[var_name]>data['q7']) & (data[var_name]<=data['q8']) , var_name + '_rank']= 8
    data.loc[(data[var_name]>data['q8']) & (data[var_name]<=data['q9']) , var_name + '_rank']= 9
    data.loc[data[var_name]>data['q9'] , var_name + '_rank']= 10

    del data['q1']
    del data['q2']
    del data['q3']
    del data['q4']
    del data['q5']
    del data['q6']
    del data['q7']
    del data['q8']
    del data['q9']

    del data[date_name+'_june']
    data =data[data[var_name + '_rank']!=0]
    return data

def get_sub_divide_rank(data, stock_name, date_name, var_name, var1_rank_name):
    df_save = pd.DataFrame()
    qs = pd.unique(data[var1_rank_name])

    for q in qs:
        data_temp = data[data[var1_rank_name]==q] 
    
        for quantile in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
            if quantile==0.1:
                quantile_table = pd.DataFrame(data_temp.groupby(date_name)[var_name].quantile(quantile)).rename(columns={var_name:str('q'+str(int(quantile*10)))})
            else: 
                quantile_table = pd.merge(quantile_table, pd.DataFrame(data_temp.groupby(date_name)[var_name].quantile(quantile)).rename(columns={var_name:str('q'+str(int(quantile*10)))}),
                                 how='left', left_index=True, right_index=True)
    
        data_temp = pd.merge(data_temp, quantile_table, how='left', left_on=date_name, right_index=True)

        data_temp[var_name+'_sub_rank'] = 0
        data_temp.loc[data_temp[var_name]<=data_temp['q1'] , var_name+'_sub_rank'] = 1
        data_temp.loc[(data_temp[var_name]>data_temp['q1']) & (data_temp[var_name]<=data_temp['q2']) , var_name+'_sub_rank'] = 2
        data_temp.loc[(data_temp[var_name]>data_temp['q2']) & (data_temp[var_name]<=data_temp['q3']) , var_name+'_sub_rank'] = 3
        data_temp.loc[(data_temp[var_name]>data_temp['q3']) & (data_temp[var_name]<=data_temp['q4']) , var_name+'_sub_rank'] = 4
        data_temp.loc[(data_temp[var_name]>data_temp['q4']) & (data_temp[var_name]<=data_temp['q5']) , var_name+'_sub_rank'] = 5
        data_temp.loc[(data_temp[var_name]>data_temp['q5']) & (data_temp[var_name]<=data_temp['q6']) , var_name+'_sub_rank'] = 6
        data_temp.loc[(data_temp[var_name]>data_temp['q6']) & (data_temp[var_name]<=data_temp['q7']) , var_name+'_sub_rank'] = 7
        data_temp.loc[(data_temp[var_name]>data_temp['q7']) & (data_temp[var_name]<=data_temp['q8']) , var_name+'_sub_rank'] = 8
        data_temp.loc[(data_temp[var_name]>data_temp['q8']) & (data_temp[var_name]<=data_temp['q9']) , var_name+'_sub_rank'] = 9
        data_temp.loc[data_temp[var_name]>data_temp['q9'] , var_name+'_sub_rank'] = 10

        data_temp = data_temp[[stock_name,date_name,var_name+'_sub_rank']]
    #data_temp[var1_rank_name] = q
        df_save = df_save.append(data_temp)

    data = pd.merge(data,df_save,how='left',on=[stock_name,date_name])
    return data

def get_port_ret(data, var1_name, var2_name, return_name, date_name):
    dates = sorted(pd.unique(data[date_name]))
    table = pd.DataFrame()

    for date in dates:
        for var1 in sorted(pd.unique(df[var1_name])):
            for var2 in sorted(pd.unique(df[var2_name])):
                data_temp = data[(data[var1_name]==var1) & (data[var2_name]==var2) & (data[date_name]==date)]
                table_temp = pd.DataFrame([date, var1, var2, np.mean(data_temp[return_name])], 
                                      index=[date_name, var1_name, var2_name, 'port']).T
                table = table.append(table_temp)
    return table

df = pd.read_csv('df2.csv',index_col=0)
df['datadate'] = pd.to_datetime(df['datadate'])
df = df.dropna()

# merge exchange code

trading = pd.read_csv('CompustatSecurityMonthly1962-2022.csv')
trading['datadate'] = pd.to_datetime(trading['datadate'])

trading = trading[(trading['exchg']==11) |
                 (trading['exchg']==12) |
                 (trading['exchg']==14) |
                 (trading['exchg']==15) |
                 (trading['exchg']==17)]

# financial firms are excluded due to high leverage
trading = trading[~trading['sic'].isin([6021, 6022, 6029, 6035, 6036, 6099, 6111,
                             6141, 6153, 6159, 6162, 6163, 6172, 6189,
                             6199, 6200, 6211, 6221, 6282, 6311, 6321,
                             6324, 6331, 6351, 6361, 6399, 6411])]

# make sure the monthend date
trading['datadate'] = trading['datadate'] + MonthEnd(0)

df = pd.merge(df, trading[['gvkey','datadate','exchg']], how='left', on=['gvkey','datadate'])
df['BE/ME'] =df['BE']/df['ME']
df['E/P'] = df['E']/df['ME']

data = df
exchange_name = 'exchg'
exchang_code = 11
var_name = 'BE/ME'
date_name = 'datadate'
stock_name = 'gvkey'

df = get_var_rank(data, exchange_name, exchang_code, var_name, date_name, stock_name)

data = df
exchange_name = 'exchg'
exchang_code = 11
var_name = 'E/P'
date_name = 'datadate'
stock_name = 'gvkey'

df = get_var_rank(data, exchange_name, exchang_code, var_name, date_name, stock_name)

df = df[df['datadate']>=pd.to_datetime('1963-1-1')]

data = df
rank_name = 'BE/ME_rank'
date_name = 'datadate'

var_name = 'trt1m'
rename = 'Return'
table = get_uni_port_value(data, rank_name, var_name, date_name, rename)

var_name = 'beta'
rename = 'beta'
table_temp = get_uni_port_value(data, rank_name, var_name, date_name, rename)
table = table.append(table_temp)

var_name = 'ln(ME)'
rename = 'ln(ME)'
table_temp = get_uni_port_value(data, rank_name, var_name, date_name, rename)
table = table.append(table_temp)

var_name = 'ln(BE/ME)'
rename = 'ln(BE/ME)'
table_temp = get_uni_port_value(data, rank_name, var_name, date_name, rename)
table = table.append(table_temp)

var_name = 'ln(A/ME)'
rename = 'ln(A/ME)'
table_temp = get_uni_port_value(data, rank_name, var_name, date_name, rename)
table = table.append(table_temp)

var_name = 'ln(A/BE)'
rename = 'ln(A/BE)'
table_temp = get_uni_port_value(data, rank_name, var_name, date_name, rename)
table = table.append(table_temp)

var_name = 'E/P dummy'
rename = 'E/P dummy'
table_temp = get_uni_port_value(data, rank_name, var_name, date_name, rename)
table = table.append(table_temp)

var_name = 'E(+)/P'
rename = 'E(+)/P'
table_temp = get_uni_port_value(data, rank_name, var_name, date_name, rename)
table = table.append(table_temp)

rename = 'Firms'
temp = data.groupby([date_name, rank_name])[var_name].count().reset_index(drop=False)
temp = pd.DataFrame(temp.groupby(rank_name)[var_name].mean()).rename(columns={var_name:rename}).T
table = table.append(temp)

table = round(table, 2)
table.to_csv('Table4A.csv')

data = df
rank_name = 'E/P_rank'
date_name = 'datadate'

var_name = 'trt1m'
rename = 'Return'
table = get_uni_port_value(data, rank_name, var_name, date_name, rename)

var_name = 'beta'
rename = 'beta'
table_temp = get_uni_port_value(data, rank_name, var_name, date_name, rename)
table = table.append(table_temp)

var_name = 'ln(ME)'
rename = 'ln(ME)'
table_temp = get_uni_port_value(data, rank_name, var_name, date_name, rename)
table = table.append(table_temp)

var_name = 'ln(BE/ME)'
rename = 'ln(BE/ME)'
table_temp = get_uni_port_value(data, rank_name, var_name, date_name, rename)
table = table.append(table_temp)

var_name = 'ln(A/ME)'
rename = 'ln(A/ME)'
table_temp = get_uni_port_value(data, rank_name, var_name, date_name, rename)
table = table.append(table_temp)

var_name = 'ln(A/BE)'
rename = 'ln(A/BE)'
table_temp = get_uni_port_value(data, rank_name, var_name, date_name, rename)
table = table.append(table_temp)

var_name = 'E/P dummy'
rename = 'E/P dummy'
table_temp = get_uni_port_value(data, rank_name, var_name, date_name, rename)
table = table.append(table_temp)

var_name = 'E(+)/P'
rename = 'E(+)/P'
table_temp = get_uni_port_value(data, rank_name, var_name, date_name, rename)
table = table.append(table_temp)

rename = 'Firms'
temp = data.groupby([date_name, rank_name])[var_name].count().reset_index(drop=False)
temp = pd.DataFrame(temp.groupby(rank_name)[var_name].mean()).rename(columns={var_name:rename}).T
table = table.append(temp)

table = round(table, 2)
table.to_csv('Table4B.csv')

data = df
return_name = 'trt1m'
stock_name ='gvkey'
date_name = 'datadate'
var_name = 'BE/ME'
var1_rank_name = 'ME_rank'

df = get_sub_divide_rank(data, stock_name, date_name, var_name, var1_rank_name)

data = df
var1_name = 'ME_rank'
var2_name = 'BE/ME_rank'
return_name = 'trt1m'
date_name = 'datadate'

table = get_port_ret(data, var1_name, var2_name, return_name, date_name)
table = table.fillna(0)

table_mean = pd.DataFrame()

for var1 in sorted(pd.unique(df[var1_name])):
    for var2 in sorted(pd.unique(df[var2_name])):
            table_temp = table[(table[var1_name]==var1) & (table[var2_name]==var2)]
            table_mean.at['ME'+str(int(var1)), 'BE/ME'+str(int(var2))] = np.mean(table_temp['port'])

table_mean = round(table_mean,2)
table_mean.to_csv('Table5.csv')

df.to_csv('df4.csv')