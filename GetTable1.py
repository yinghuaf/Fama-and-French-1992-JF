import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from pandas.tseries.offsets import *
import statsmodels.api as sm

trading = pd.read_csv('CompustatSecurityMonthly1962-2022.csv')
trading['datadate'] = pd.to_datetime(trading['datadate'])

funda = pd.read_csv('CompustatFundamantalAnnual1959-2022.csv')
funda['datadate'] = pd.to_datetime(funda['datadate'])

# filtering
# stock exchange
# 11: New York Stock Exchange
# 12: American SE
# 14: NASDAQ-NMS Stock Market
# 15: NASDAQ OMX Boston
# 17: NYSE Arca
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

# the unit for trt1m is percent
# initial
df = trading[['gvkey','cusip','sic','datadate','trt1m']]

# calculate market equity
# portfolios are formed yearly: June of year t
# breakpoints are based on the sizeusing all NYSE stocks
# all NYSE, AMEX and NASDAQ stocks are allocated into the 10 portfolios according to the breakpoints

###########################
# calculate market equity
###########################
trading_sub = trading[trading['datadate'].astype(str).str[-5:]=='06-30']
trading_sub['ME'] = trading_sub['prccm']*trading_sub['cshoq']

# break points are based on NYSE
trading_sub_temp = trading_sub[(trading_sub['exchg']==11) | (trading_sub['exchg']==17)]
trading_sub_temp = trading_sub_temp[trading_sub_temp['ME'].notna()]

# seeting june
trading_sub['datadate_june'] = trading_sub['datadate']

trading = pd.merge(trading, trading_sub[['datadate','datadate_june']].drop_duplicates(), how='left', on='datadate')
trading = trading.sort_values(by=['gvkey','datadate'])

temp = pd.DataFrame(trading.groupby('gvkey')['datadate_june'].fillna(method='ffill'))
del trading['datadate_june']

trading = pd.merge(trading, temp, how='left', left_index=True, right_index=True)

# create quantile table
for quantile in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
    if quantile==0.1:
        quantile_table = pd.DataFrame(trading_sub_temp.groupby('datadate')['ME'].quantile(quantile)).rename(columns={'ME':str('q'+str(int(quantile*10)))})
    else: 
        quantile_table = pd.merge(quantile_table, pd.DataFrame(trading_sub_temp.groupby('datadate')['ME'].quantile(quantile)).rename(columns={'ME':str('q'+str(int(quantile*10)))}),
                                 how='left', left_index=True, right_index=True)

#trading = pd.merge(trading, quantile_table, how='left', left_on='datadate_june', right_index=True)
trading = pd.merge(trading, trading_sub[['gvkey','datadate_june','ME']].drop_duplicates(),
         how='left', on=['gvkey','datadate_june'])

trading = trading[trading['ME'].notna()]

trading = pd.merge(trading, quantile_table, how='left', left_on='datadate_june', right_index=True)

trading['ME_rank'] = 0
trading.loc[trading['ME']<=trading['q1'] , 'ME_rank']= 1
trading.loc[(trading['ME']>trading['q1']) & (trading['ME']<=trading['q2']) , 'ME_rank']= 2
trading.loc[(trading['ME']>trading['q2']) & (trading['ME']<=trading['q3']) , 'ME_rank']= 3
trading.loc[(trading['ME']>trading['q3']) & (trading['ME']<=trading['q4']) , 'ME_rank']= 4
trading.loc[(trading['ME']>trading['q4']) & (trading['ME']<=trading['q5']) , 'ME_rank']= 5
trading.loc[(trading['ME']>trading['q5']) & (trading['ME']<=trading['q6']) , 'ME_rank']= 6
trading.loc[(trading['ME']>trading['q6']) & (trading['ME']<=trading['q7']) , 'ME_rank']= 7
trading.loc[(trading['ME']>trading['q7']) & (trading['ME']<=trading['q8']) , 'ME_rank']= 8
trading.loc[(trading['ME']>trading['q8']) & (trading['ME']<=trading['q9']) , 'ME_rank']= 9
trading.loc[trading['ME']>trading['q9'] , 'ME_rank']= 10

#trading = pd.merge(trading, trading_sub[['gvkey','datadate_june','ME_rank']], how='left', on=['gvkey','datadate_june'])

# merge ME and rank of ME with df
df = pd.merge(df, trading[['gvkey','datadate','ME','ME_rank']], how='left', on=['gvkey','datadate'])
df = df[df['ME'].notna()]

df = df.reset_index(drop=True)

###########################
# calculate pre-ranking beta
###########################

# calculate market
# with the CRSP value-weighted portfolio of NYSE, AMEX, 
# and (after 1972) NASDAQ stocks used as the proxy for the market.
temp = trading[(trading['datadate']<=pd.to_datetime('1972-12-31')) & (trading['exchg'].isin([11,12,17]))]
temp = temp.groupby('datadate').apply(lambda x: np.sum(x['ME']/np.sum(x['ME'])*x['trt1m'])).reset_index(drop=False)
temp = temp.rename(columns={0:'mkt'})

temp1 = trading[trading['datadate']>pd.to_datetime('1972-12-31')]
temp1 = temp1.groupby('datadate').apply(lambda x: np.sum(x['ME']/np.sum(x['ME'])*x['trt1m'])).reset_index(drop=False)
temp1 = temp1.rename(columns={0:'mkt'})

temp = temp.append(temp1)

trading = pd.merge(trading, temp, how='left', on='datadate')

rolling_count = 24
df_beta = pd.DataFrame()
dates = sorted(pd.unique(trading['datadate']))

for date in dates[24:]:
    date = pd.to_datetime(date)
    if date != str(date)[5:10] != '06-30':
        continue
        
    if rolling_count <= 60:
        trading_temp = trading[trading['datadate']<=date]
    else:
        trading_temp = trading[(trading['datadate']<=date) & (trading['datadate']>=(date-MonthEnd(60)))]
        
    rolling_count = rolling_count + 1  
    for me_rank in range(1,11):
        trading_temp_temp = trading_temp[trading_temp['ME_rank']==me_rank]
        
        for quantile in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
            if quantile==0.1:
                quantile_table = pd.DataFrame(trading_temp_temp.groupby('datadate')['ME'].quantile(quantile)).rename(columns={'ME':str('qq'+str(int(quantile*10)))})
            else: 
                quantile_table = pd.merge(quantile_table, pd.DataFrame(trading_temp_temp.groupby('datadate')['ME'].quantile(quantile)).rename(columns={'ME':str('qq'+str(int(quantile*10)))}),
                                 how='left', left_index=True, right_index=True)
                
        trading_temp_temp = pd.merge(trading_temp_temp, quantile_table, how='left', left_on='datadate', right_index=True)
        trading_temp_temp['ME_rank_q'] = 0
        
        trading_temp_temp.loc[(trading_temp_temp['ME']<=trading_temp_temp['qq1']), 'ME_rank_q']= 1
        trading_temp_temp.loc[(trading_temp_temp['ME']>trading_temp_temp['qq1']) & (trading_temp_temp['ME']<=trading_temp_temp['qq2']), 'ME_rank_q']= 2
        trading_temp_temp.loc[(trading_temp_temp['ME']>trading_temp_temp['qq2']) & (trading_temp_temp['ME']<=trading_temp_temp['qq3']), 'ME_rank_q']= 3
        trading_temp_temp.loc[(trading_temp_temp['ME']>trading_temp_temp['qq3']) & (trading_temp_temp['ME']<=trading_temp_temp['qq4']), 'ME_rank_q']= 4
        trading_temp_temp.loc[(trading_temp_temp['ME']>trading_temp_temp['qq4']) & (trading_temp_temp['ME']<=trading_temp_temp['qq5']), 'ME_rank_q']= 5
        trading_temp_temp.loc[(trading_temp_temp['ME']>trading_temp_temp['qq5']) & (trading_temp_temp['ME']<=trading_temp_temp['qq6']), 'ME_rank_q']= 6
        trading_temp_temp.loc[(trading_temp_temp['ME']>trading_temp_temp['qq6']) & (trading_temp_temp['ME']<=trading_temp_temp['qq7']), 'ME_rank_q']= 7
        trading_temp_temp.loc[(trading_temp_temp['ME']>trading_temp_temp['qq7']) & (trading_temp_temp['ME']<=trading_temp_temp['qq8']), 'ME_rank_q']= 8
        trading_temp_temp.loc[(trading_temp_temp['ME']>trading_temp_temp['qq8']) & (trading_temp_temp['ME']<=trading_temp_temp['qq9']), 'ME_rank_q']= 9
        trading_temp_temp.loc[(trading_temp_temp['ME']>trading_temp_temp['qq9']), 'ME_rank_q'] = 10
        
        for me_rank in range(1,11):
            trading_temp_temp_temp = trading_temp_temp[trading_temp_temp['ME_rank_q']==me_rank]
            
            OLS_temp = pd.DataFrame(trading_temp_temp_temp.groupby('datadate').apply(lambda x: np.sum(x['ME']/np.sum(x['ME'])*x['trt1m']))).rename(columns={0:'port'})
            OLS_temp = pd.merge(OLS_temp, trading_temp_temp_temp[['datadate','mkt']].drop_duplicates(),
                 how='left', left_index=True, right_on='datadate')
            OLS_temp['const'] = 1
            
            if len(OLS_temp) <= 1:
                continue
            
            result = sm.OLS(OLS_temp['port'],
               OLS_temp[['const','mkt']], missing='drop').fit()

            df_beta_temp = pd.DataFrame(trading_temp_temp_temp['gvkey'])
            df_beta_temp['beta'] = result.params[1]
            df_beta_temp['datadate'] = date
    
            df_beta = df_beta.append(df_beta_temp)

df_beta = df_beta.drop_duplicates()

# portfolio sorting
trading_sub = trading[(trading['exchg']==11) | (trading['exchg']==17)]
trading_sub = pd.merge(trading_sub, df_beta, how='left', on=['gvkey','datadate'])

trading_sub = trading_sub.sort_values(by=['gvkey','datadate'])
temp = pd.DataFrame(trading_sub.groupby('gvkey')['beta'].fillna(method='ffill'))
del trading_sub['beta']
trading_sub = pd.merge(trading_sub, temp, how='left', left_index=True, right_index=True)

# create quantile table
for quantile in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
    if quantile==0.1:
        quantile_table = pd.DataFrame(trading_sub.groupby('datadate')['beta'].quantile(quantile)).rename(columns={'beta':str('q'+str(int(quantile*10)))})
    else: 
        quantile_table = pd.merge(quantile_table, pd.DataFrame(trading_sub.groupby('datadate')['beta'].quantile(quantile)).rename(columns={'beta':str('q'+str(int(quantile*10)))}),
                                 how='left', left_index=True, right_index=True)

trading = pd.merge(trading, df_beta, how='left', on=['gvkey','datadate'])

temp = pd.DataFrame(trading.groupby('gvkey')['beta'].fillna(method='ffill'))
del trading['beta']
trading = pd.merge(trading, temp, how='left', left_index=True, right_index=True)

del trading['q1']
del trading['q2']
del trading['q3']
del trading['q4']
del trading['q5']
del trading['q6']
del trading['q7']
del trading['q8']
del trading['q9']

trading = pd.merge(trading, quantile_table, how='left', left_on='datadate', right_index=True)

trading['beta_rank'] = 0
trading.loc[trading['beta']<=trading['q1'] , 'beta_rank'] = 1
trading.loc[(trading['beta']>trading['q1']) & (trading['beta']<=trading['q2']) , 'beta_rank'] = 2
trading.loc[(trading['beta']>trading['q2']) & (trading['beta']<=trading['q3']) , 'beta_rank'] = 3
trading.loc[(trading['beta']>trading['q3']) & (trading['beta']<=trading['q4']) , 'beta_rank'] = 4
trading.loc[(trading['beta']>trading['q4']) & (trading['beta']<=trading['q5']) , 'beta_rank'] = 5
trading.loc[(trading['beta']>trading['q5']) & (trading['beta']<=trading['q6']) , 'beta_rank'] = 6
trading.loc[(trading['beta']>trading['q6']) & (trading['beta']<=trading['q7']) , 'beta_rank'] = 7
trading.loc[(trading['beta']>trading['q7']) & (trading['beta']<=trading['q8']) , 'beta_rank'] = 8
trading.loc[(trading['beta']>trading['q8']) & (trading['beta']<=trading['q9']) , 'beta_rank'] = 9
trading.loc[trading['beta']>trading['q9'] , 'beta_rank'] = 10

# merge beta and rank of beta with df
df = pd.merge(df, trading[['gvkey','datadate','beta','beta_rank']], how='left', on=['gvkey','datadate'])

df = df[df['beta_rank']!=0]
df = df[df['beta'].notna()]

df = df[~df[['gvkey','datadate']].duplicated()]
df = df.reset_index(drop=True)

###########################
# calculate post-ranking beta
###########################
# we calculate the equal-weighted monthly returns on the portfolios 
# for the next 12 months, from July to June.

temp_beta = df.groupby(['datadate','ME_rank','beta_rank']).apply(lambda x: np.sum(x['trt1m']/len(x['trt1m']))).reset_index(drop=False)
temp_beta = temp_beta.rename(columns={0:'port'})

# calcualte market portfolio return
temp = trading[(trading['datadate']<=pd.to_datetime('1972-12-31')) & (trading['exchg'].isin([11,12,17]))]
temp = temp.groupby('datadate').apply(lambda x: np.sum(x['ME']/np.sum(x['ME'])*x['trt1m'])).reset_index(drop=False)
temp = temp.rename(columns={0:'mkt'})

temp1 = trading[trading['datadate']>pd.to_datetime('1972-12-31')]
temp1 = temp1.groupby('datadate').apply(lambda x: np.sum(x['ME']/np.sum(x['ME'])*x['trt1m'])).reset_index(drop=False)
temp1 = temp1.rename(columns={0:'mkt'})

temp = temp.append(temp1)

temp_beta = pd.merge(temp_beta, temp, how='left', on='datadate')
temp_beta['const'] = 1

df_beta = pd.DataFrame()

for me in range(1,11):
    for beta in range(1,11):
        temp_beta_temp = temp_beta[(temp_beta['ME_rank']==me) & (temp_beta['beta_rank']==beta)]
        
        result = sm.OLS(temp_beta_temp['port'],
               temp_beta_temp[['const','mkt']], missing='drop').fit()
        
        df_beta_temp = pd.DataFrame([me,beta,result.params[1]], index=['ME_rank','beta_rank','post_beta']).T
        
        df_beta = df_beta.append(df_beta_temp)

trading_sub = trading[(trading['exchg']==11) | (trading['exchg']==17)]
trading_sub = pd.merge(trading_sub, df_beta, how='left', on=['ME_rank','beta_rank'])

# create quantile table
for quantile in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
    if quantile==0.1:
        quantile_table = pd.DataFrame(trading_sub.groupby('datadate')['post_beta'].quantile(quantile)).rename(columns={'post_beta':str('q'+str(int(quantile*10)))})
    else: 
        quantile_table = pd.merge(quantile_table, pd.DataFrame(trading_sub.groupby('datadate')['post_beta'].quantile(quantile)).rename(columns={'post_beta':str('q'+str(int(quantile*10)))}),
                                 how='left', left_index=True, right_index=True)

trading = pd.merge(trading, df_beta, how='left', on=['ME_rank','beta_rank'])

del trading['q1']
del trading['q2']
del trading['q3']
del trading['q4']
del trading['q5']
del trading['q6']
del trading['q7']
del trading['q8']
del trading['q9']

trading = pd.merge(trading, quantile_table, how='left', left_on='datadate', right_index=True)

trading['post_beta_rank'] = 0
trading.loc[trading['post_beta']<=trading['q1'] , 'post_beta_rank'] = 1
trading.loc[(trading['post_beta']>trading['q1']) & (trading['post_beta']<=trading['q2']) , 'post_beta_rank'] = 2
trading.loc[(trading['post_beta']>trading['q2']) & (trading['post_beta']<=trading['q3']) , 'post_beta_rank'] = 3
trading.loc[(trading['post_beta']>trading['q3']) & (trading['post_beta']<=trading['q4']) , 'post_beta_rank'] = 4
trading.loc[(trading['post_beta']>trading['q4']) & (trading['post_beta']<=trading['q5']) , 'post_beta_rank'] = 5
trading.loc[(trading['post_beta']>trading['q5']) & (trading['post_beta']<=trading['q6']) , 'post_beta_rank'] = 6
trading.loc[(trading['post_beta']>trading['q6']) & (trading['post_beta']<=trading['q7']) , 'post_beta_rank'] = 7
trading.loc[(trading['post_beta']>trading['q7']) & (trading['post_beta']<=trading['q8']) , 'post_beta_rank'] = 8
trading.loc[(trading['post_beta']>trading['q8']) & (trading['post_beta']<=trading['q9']) , 'post_beta_rank'] = 9
trading.loc[trading['post_beta']>trading['q9'] , 'post_beta_rank'] = 10

trading = trading[~trading[['gvkey','datadate']].duplicated()]

# merge beta and rank of beta with df
df = pd.merge(df, trading[['gvkey','datadate','post_beta','post_beta_rank']], how='left', on=['gvkey','datadate'])

df = df[df['post_beta_rank']!=0]

df = df[df['trt1m'].notna()]
df = df.reset_index(drop=True)

df.to_csv('df1.csv')

df = pd.read_csv('df1.csv', index_col=0)
df['datadate'] = pd.to_datetime(df['datadate'])



df = df[df['datadate']>=pd.to_datetime('1963-1-1')]

###########################
# Table 1 replication
###########################

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

# sub divide
data = df
return_name = 'trt1m'
stock_name ='gvkey'
date_name = 'datadate'
var_name = 'beta'
var1_rank_name = 'ME_rank'

df = get_sub_divide_rank(data, stock_name, date_name, var_name, var1_rank_name)

data = df
return_name = 'trt1m'
stock_name ='gvkey'
date_name = 'datadate'
var_name = 'post_beta'
var1_rank_name = 'ME_rank'

df = get_sub_divide_rank(data, stock_name, date_name, var_name, var1_rank_name)

ef get_port_ret(data, var1_name, var2_name, return_name, date_name):
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

data = df
var1_name = 'ME_rank'
var2_name = 'beta_rank'
return_name = 'trt1m'
date_name = 'datadate'

table = get_port_ret(data, var1_name, var2_name, return_name, date_name)
table = table.fillna(0)

table_mean = pd.DataFrame()

for var1 in sorted(pd.unique(df[var1_name])):
    for var2 in sorted(pd.unique(df[var2_name])):
            table_temp = table[(table[var1_name]==var1) & (table[var2_name]==var2)]
            table_mean.at['ME'+str(int(var1)), 'Beta'+str(int(var2))] = np.mean(table_temp['port'])

table_mean = round(table_mean,2)
table_mean.to_csv('Table1A.csv')

data = df
var1_name = 'ME_rank'
var2_name = 'beta_rank'
return_name = 'post_beta'
date_name = 'datadate'

table = get_port_ret(data, var1_name, var2_name, return_name, date_name)
table = table.fillna(0)

table_mean = pd.DataFrame()

for var1 in sorted(pd.unique(df[var1_name])):
    for var2 in sorted(pd.unique(df[var2_name])):
            table_temp = table[(table[var1_name]==var1) & (table[var2_name]==var2)]
            table_mean.at['ME'+str(int(var1)), 'Beta'+str(int(var2))] = np.mean(table_temp['port'])

table_mean = round(table_mean,2)
table_mean.to_csv('Table1B.csv')

df.loc[df['ME']<=0,'ME']=1
df['me'] = np.log(df['ME']*1)
data = df
var1_name = 'ME_rank'
var2_name = 'beta_rank'
return_name = 'me'
date_name = 'datadate'

table = get_port_ret(data, var1_name, var2_name, return_name, date_name)
table = table.fillna(0)

table_mean = pd.DataFrame()

for var1 in sorted(pd.unique(df[var1_name])):
    for var2 in sorted(pd.unique(df[var2_name])):
            table_temp = table[(table[var1_name]==var1) & (table[var2_name]==var2)]
            table_mean.at['ME'+str(int(var1)), 'Beta'+str(int(var2))] = np.mean(table_temp['port'])

table_mean = round(table_mean,2)
table_mean.to_csv('Table1C.csv')