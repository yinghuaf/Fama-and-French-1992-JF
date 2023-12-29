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

trading = pd.read_csv('CompustatSecurityMonthly1962-2022.csv')
trading['datadate'] = pd.to_datetime(trading['datadate'])

funda = pd.read_csv('CompustatFundamantalAnnual1959-2022.csv')
funda['datadate'] = pd.to_datetime(funda['datadate'])

df = pd.read_csv('df1.csv', index_col=0)
df['datadate'] = pd.to_datetime(df['datadate'])

# fill nan by 0
for var in ['seq','txditc','at','ib','dvpa','txdb']:
    funda[var] = funda[var].fillna(0)

# ln(BE/ME)
funda['BE'] = funda['seq'] + funda['txditc']
#funda.loc[funda['BE'] < 0, 'BE'] = 0

# total assets
funda['A'] = funda['at']
funda['E'] = funda['ib'] + funda['txdb'] - funda['dvpa'] 

# june date
funda['datadate_june'] = pd.to_datetime(funda['datadate'].astype(str).str[:-5] + '06-30')

funda.loc[funda['datadate']>funda['datadate_june'], 'datadate_june'] = pd.to_datetime((funda[funda['datadate']>funda['datadate_june']]['datadate_june'].astype(str).str[:-6].astype(int)+1).astype(str) + '-06-30')

df = pd.merge(df, funda[['gvkey','datadate_june','BE','A','E']], how='left', left_on=['gvkey','datadate'], right_on=['gvkey','datadate_june'])

for var in ['BE','A','E']:
    data = df
    var_name = var
    date_name = 'datadate'
    stock_name = 'gvkey'
    df = get_ffill(data, var_name, date_name, stock_name)

del df['datadate_june']

# 
df['ln(ME)'] = np.log(df['ME'])
df['ln(BE/ME)'] = np.log(df['BE']/df['ME'])
df['ln(A/ME)'] = np.log(df['A']/df['ME'])
df['ln(A/BE)'] = np.log(df['A']/df['BE'])

df['E/P dummy'] = 0
df.loc[df['E']<0, 'E/P dummy'] = 1

df['E(+)'] = df['E']
df.loc[df['E']<0, 'E(+)'] = 0
df['E(+)/P'] = df['E(+)']/df['ME']

data = df
rank_name = 'ME_rank'
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
temp = pd.DataFrame(temp.groupby('ME_rank')[var_name].mean()).rename(columns={var_name:rename}).T
table = table.append(temp)

table = round(table, 2)
table.to_csv('Table2A.csv')

data = df
rank_name = 'ME_rank'
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
temp = pd.DataFrame(temp.groupby('ME_rank')[var_name].mean()).rename(columns={var_name:rename}).T
table = table.append(temp)

table = round(table, 2)
table.to_csv('Table2A.csv')

data = df
rank_name = 'beta_rank'
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
table.to_csv('Table2B.csv')

df.to_csv('df2.csv')