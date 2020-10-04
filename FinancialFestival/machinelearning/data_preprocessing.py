import numpy as np
import pandas as pd
import datetime
from sklearn.preprocessing import StandardScaler

stocks = pd.read_csv("../data/stocks.csv")
len_stocks = 286061 
trade = pd.read_csv("../data/trade_train.csv")

#append date
# 20190701 -> 201907, 1
def datesep_d(num):
    return num % 100
def datesep_ym(num):
    return num // 100

dic = {'기준일자':'date'}
df_temp = stocks[dic.keys()].rename(columns=dic)
df_temp['date_ym'] = df_temp.apply(lambda func: datesep_ym(func['date']), axis = 1)
df_temp['date_d'] = df_temp.apply(lambda func: datesep_d(func['date']), axis = 1)
df = df_temp[['date_ym','date_d']]

#append stock_code
dic = {'종목번호':'stock_code'}
df = df.join(stocks[dic.keys()].rename(columns=dic))

#append kospi
df_temp = pd.get_dummies(stocks['시장구분'])
df_temp.columns=['kosdaq','kospi']
df = df.join(df_temp)

#append standard1
df_temp = pd.get_dummies(stocks['표준산업구분코드_대분류'])
std1_col = df_temp.columns
col_temp = []
for i in range(14):
    exec('col_temp.append("std1_{}")'.format(i))
df_temp.columns = col_temp
df = df.join(df_temp)

#append standard2
df_temp = pd.get_dummies(stocks['표준산업구분코드_중분류'])
std2_col = df_temp.columns
col_temp = []
for i in range(56):
    exec('col_temp.append("std2_{}")'.format(i))
df_temp.columns = col_temp
df = df.join(df_temp)

#append standard3
df_temp = pd.get_dummies(stocks['표준산업구분코드_소분류'])
std3_col = df_temp.columns
col_temp = []
for i in range(117):
    exec('col_temp.append("std3_{}")'.format(i))
df_temp.columns = col_temp
df = df.join(df_temp)
#original col values: (std1_col, std2_col, std3_col)

#append price_low2high, start2end
def pricescaler(num1, num2):
    # at low2high 0->0, 100->100 consider the same case
    interval = num2 - num1
    if num1 == 0: return 0
    else: return interval/num1

dic = {'종목저가':'low', '종목고가':'high','종목시가':'start','종목종가':'end'}
df_temp = stocks[dic.keys()].rename(columns=dic)
df_temp['price_low2high'] = df_temp.apply(lambda func: pricescaler(func['low'], func['high']), axis=1)
df_temp['price_start2end'] = df_temp.apply(lambda func: pricescaler(func['start'], func['end']), axis=1)
df = df.join(df_temp[['price_low2high','price_start2end']])

#append trade_volume, trade_amount
dic = {'거래량':'trade_volume','거래금액_만원단위':'trade_amount'}
df_temp = stocks[dic.keys()].rename(columns=dic)
std_scaler = StandardScaler()
fitted = std_scaler.fit(df_temp)
df_temp = std_scaler.transform(df_temp)
df_temp = pd.DataFrame(df_temp, columns=dic.values())
df = df.join(df_temp)

#target 2020.07 top3 stock_code
def findtarget(target, check):
    if (check == 'Y'):
        return target
dic = {'종목번호':'stock_code','20년7월TOP3대상여부':'targetcheck'}
target = stocks[dic.keys()].rename(columns=dic)
target['target'] = target.apply(lambda func: findtarget(func['stock_code'], func['targetcheck']), axis=1)
target = target.drop(['stock_code','targetcheck'], axis=1)
target = target.dropna(axis=0)
target = target.drop_duplicates('target', keep='first')
target = target.reset_index(drop=True)

#concat with trade_train
def getlabel(buy_num, group_size):
    return buy_num / group_size

dic = {'그룹번호':'group_idx','기준년월':'date_ym','종목번호':'stock_code','매수고객수':'buy_num','그룹내고객수':'group_size'}
trade = trade[dic.keys()].rename(columns=dic)
trade = trade.sort_values(by=['group_idx','date_ym'])
trade['label'] = trade.apply(lambda func: getlabel(func['buy_num'], func['group_size']), axis=1)
trade = trade.drop(['buy_num','group_size'], axis=1)

#split trade dataframe by group_idx
group_idx = trade['group_idx'].tolist()
group_idx = sorted(list(set(group_idx)))
df_list = []
for idx in group_idx:
    exec('df_{} = trade[trade["group_idx"] == idx]'.format(idx))
    exec('df_{} = df_{}.reset_index(drop=True)'.format(idx, idx))
    exec('df_{} = df_{}.drop(["group_idx"], axis=1)'.format(idx, idx))
    exec('df_list.append(df_{})'.format(idx))
#df_list => list of df_MAD01, ...

#make test set
test = df[ (df["stock_code"]==target.iloc[0]["target"]) & (df["date_ym"] == 202007) ]
for target_idx in range(1, len(target)):
    target_temp = target.iloc[target_idx]['target']
    test_temp = df[ (df['stock_code']==target_temp) & (df['date_ym'] == 202007) ]
    test = pd.concat([test, test_temp])
test = test.drop(['date_ym','date_d'], axis=1)
test.set_index('stock_code', inplace=True)

#make train set
# df_list -> date_ym, stock_code, label 
train_list = []
label_list = []
for idx in group_idx:
    exec('train_{} = df[ (df["stock_code"]==df_{}.iloc[0]["stock_code"]) & (df["date_ym"] == df_{}.iloc[0]["date_ym"]) ]'.format(idx,idx,idx))
    exec('label_value = np.repeat( np.array(df_{}.iloc[0]["label"]), len(train_{}) )'.format(idx,idx,idx))
    exec('label_{} = pd.DataFrame(label_value, columns=["label"])'.format(idx,idx))

for i, idx in enumerate(group_idx):
    for df_i in range(1, len(df_list[i])):
        exec('code_temp = df_{}.iloc[df_i]["stock_code"]'.format(idx))
        exec('date_temp = df_{}.iloc[df_i]["date_ym"]'.format(idx))
        train_temp = df[ (df['stock_code'] == code_temp) & (df['date_ym'] == date_temp) ]
        exec('train_{} = pd.concat([train_{}, train_temp])'.format(idx, idx))
        exec('label_temp = pd.DataFrame( np.repeat(np.array(df_{}.iloc[0]["label"]), len(train_temp)), columns=["label"] )'.format(idx,idx,idx))
        exec('label_{} = pd.concat([label_{}, label_temp])'.format(idx,idx))
    exec('train_{} = train_{}.drop(["date_ym","date_d"], axis=1)'.format(idx, idx))
    exec('train_{}.set_index(train_{}["stock_code"], inplace=True)'.format(idx,idx))
    exec('train_list.append(train_{})'.format(idx))
    exec('label_list.append(label_{})'.format(idx))

#dataframe to csv
#test
test.to_csv('../data/preprocessing/test.csv')
#train, label
for i, idx in enumerate(group_idx):
    exec('train_{}.to_csv("../data/preprocessing/train/train_{}.csv")'.format(idx,idx))
    exec('label_{}.to_csv("../data/preprocessing/label/label_{}.csv")'.format(idx,idx))
