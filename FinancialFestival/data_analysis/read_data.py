import pandas as pd

stocks = pd.read_csv("../data/stocks.csv")
trade_train = pd.read_csv("../data/trade_train.csv")

pd.set_option('display.max_row',500)
pd.set_option('display.max_columns',15)
#print(stocks.columns)
#print(stocks)
#print(trade_train.columns)
#print(trade_train)

#analysis trade_train
#columns 'Unnamed: 0', '기준년월', '그룹번호', '그룹내고객수', '종목번호', '그룹내_매수여부', '그룹내_매도여부', '매수고객수', '매도고객수', '평균매수수량', '평균매도수량', '매수가격_중앙값', '매도가격_중앙값'] 

print(trade_train.columns)

for i in range(len(trade_train.columns)):
    col_name = trade_train.columns[i]
    print(col_name, len(trade_train[col_name].value_counts()))
    print(trade_train[col_name].value_counts().sort_index())
exit()


#anaysis stocks
# ['index', '기준일자', '종목번호', '종목명', '20년7월TOP3대상여부', '시장구분', '표준산업구분코드_대분류', '표준산업구분코드_중분류', '표준산업구분코드_소분류', '종목시가', '종목고가', '종목저가', '종목종가', '거래량', '거래금액_만원단위']

for i in range(len(stocks.columns)):
    col_name = stocks.columns[i]
    print(stocks[col_name].value_counts())

#print(stocks["표준산업구분코드_대분류"].value_counts())


