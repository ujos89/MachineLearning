import pandas as pd

df = pd.read_csv('./train/train.csv')
df_test = pd.read_csv('./test/0.csv')
'''
explain columns
['Day', 'Hour', 'Minute', 'DHI', 'DNI', 'WS', 'RH', 'T', 'TARGET']
DHI: diffuse horizontal irradiance W/m2
DNI: Direct Normla irradiance W/m2
WS: Wind Speed m/s
RH: Relative Humidity %
T: Temperature
Target: Solar power generate kW
print(len(df), df.columns)
print(df.loc[df['Day']==0])
'''
print(df_test)

