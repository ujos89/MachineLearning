import pandas as pd
import datetime
from sklearn.preprocessing import OneHotEncoder

stocks = pd.read_csv("../data/stocks.csv")
len_stocks = 286061 
trade = pd.read_csv("../data/trade_train.csv")

#append stock_code
dic = {'종목번호':'stock_code'}
df = stocks[dic.keys()].rename(columns=dic)

#append date
def num2date(num):
    #standard 20190701 => 0
    d = num % 100
    m = (num // 100) % 100
    y = num // 10000
    date = datetime.date(y, m ,d) - datetime.date(2019,7,1)
    return date.days

dic = {'기준일자':'date'}
df_temp = stocks[dic.keys()].rename(columns=dic)
df_temp['date'] = df_temp.apply(lambda func: num2date(func['date']), axis = 1)
df = df.join(df_temp)

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
print(df)

#original col values: (std1_col, std2_col, std3_col)
