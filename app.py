import ccxt
import pandas as pd
pd.set_option("display.precision", 8)
from datetime import  datetime
import datetime as dt
import pandas_ta as ta
from time import sleep
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
# sns.set_style("whitegrid")

class Run_model :
    def __init__(self , ex='deribit'):
        self.ex = ex
        self.pair_data = "TOMO-PERP"
        self.pair_trade = 'ETH-PERPETUAL'
        self.apiKey ="AtdG0K3k"
        self.secret ="lItUXWckP2PNN-uPnrP_h_0dsctCXdFVP9x73bwo3Nc"
        
        self.W_111 = 0.00 ;self.W_112 = 0.00 #;self.W_113 = 0.00
        self.W_121 = 0.00 ;self.W_122 = 0.00 #;self.W_123 = 0.00
#         self.W_211 = 0.00 ;self.W_212 = 0.00
#         self.W_221 = 0.00 ;self.W_222 = 0.00
        
        self.start_capital = 225.00
        self.sleep = 3
        self.timeframe = "1h"  
        self.limit = 500
        self.start_test = dt.datetime(2020, 7 , 4 , 0 , 0)
        self.length_1 = 20
        self.length_2 = 40
#         self.length_3 = 60
        self.input_1  = 'rsi'
        self.input_2  = 'rsi'
#         self.input_3  = 'rsi'
        
    @property
    def ex_api (self):
        if self.ex == "ftx" :
            exchange = ccxt.ftx({'apiKey': self.apiKey ,'secret': self.secret  , 'enableRateLimit': True }) 
        elif self.ex == "deribit":
            exchange = ccxt.deribit({'apiKey': self.apiKey,'secret': self.secret,'enableRateLimit': True,
                            "urls": {"api": "https://test.deribit.com"}})
        return exchange
        
    @property
    def dataset (self):
        self.exchange = ccxt.ftx({'apiKey': '' ,'secret': ''  , 'enableRateLimit': True }) 
        timeframe = self.timeframe 
        limit =  self.limit 
        ohlcv = self.exchange.fetch_ohlcv(self.pair_data,timeframe , limit=limit )
        ohlcv = self.exchange.convert_ohlcv_to_trading_view(ohlcv)
        df =  pd.DataFrame(ohlcv)
        df.t = df.t.apply(lambda  x :  datetime.fromtimestamp(x))
        df = df[df.t > self.start_test]
        df =  df.set_index(df['t']) ; df = df.drop(['t'] , axis= 1 )
        df = df.rename(columns={"o": "open", "h": "high"  , "l": "low", "c": "close" , "v": "volume"})
        dataset = df  ; dataset = dataset.dropna()
        return dataset

    @property  
    def talib (self): # ตัวแปร
        dataset = self.dataset
        dataset.ta.ohlc4(append=True)
        dataset['input_1'] = dataset.ta(kind=self.input_1 , length= self.length_1 , scalar=1 , append=False)
        dataset['input_2'] = dataset.ta(kind=self.input_2 , length= self.length_2 , scalar=1 , append=False)   
#         dataset['input_3'] = dataset.ta(kind=self.input_3 , length= self.length_3 , scalar=1 , append=False)   
        dataset = dataset.fillna(0)
        dataset['y_Reg'] = dataset['OHLC4'].shift(-1).fillna(dataset.OHLC4[-1])
        X = dataset.iloc[ : , 1:-1]  ;  y_Reg = dataset.iloc[ : ,[ -1]] 
        return X , y_Reg , dataset
        
    def softmax(self , x):
        xo = np.array([x.Dense_11 ,x.Dense_12])
        e_x     = np.exp(xo - np.max(xo))
        output  = e_x / e_x.sum()
        ax      =  np.argmax(output)
        return  ax
        
    @property  
    def deep (self):
        _,_, dataset = self.talib 
        dataset['Dense_11']  =  dataset.apply((lambda x : max(0, ((self.W_111 * x.input_1)+(self.W_112 * x.input_2 + 0))) , axis=1)
        dataset['Dense_12']  =  dataset.apply((lambda x : max(0, ((self.W_121 * x.input_1)+(self.W_122 * x.input_2)+ 0))) , axis=1)
#         dataset['Dense_21']  =  dataset.apply((lambda x : max(0, ((self.W_211 * x.Dense_11)+(self.W_212 * x.Dense_12)+ 0))) , axis=1)
#         dataset['Dense_22']  =  dataset.apply((lambda x : max(0, ((self.W_221 * x.Dense_11)+(self.W_222 * x.Dense_12)+0))) , axis=1)
        dataset['Output']   =  dataset.apply((lambda x : self.softmax(x)) , axis=1)
        dataset['Predict']  =  dataset['Output'] == 1
        dataset = dataset.dropna()
        return dataset
    
    @property 
    def chart (self):
        dataset = self.deep
        dataset['buy'] =  dataset.apply(lambda x : np.where( x.Predict == True , x.OHLC4 , None) , axis=1)
        dataset['sell'] = dataset.apply(lambda x : np.where( x.Predict == False, x.OHLC4 , None) , axis=1)
        plt.figure(figsize=(12,8))
        plt.plot(dataset.OHLC4 , color='k' , alpha=0.20 )
        plt.plot(dataset.buy , 'o',  color='g' , alpha=0.50 )
        plt.plot(dataset.sell , 'o', color='r' , alpha=0.50)       
        st.write('Predict:' , dataset.Predict[-1])
        st.pyplot()

    @property 
    def nav (self):
        nav_dataset = self.deep
        nav_dataset['Next_Returns'] = np.log(nav_dataset['OHLC4']/nav_dataset['OHLC4'].shift(1))
        nav_dataset['Next_Returns'] = nav_dataset['Next_Returns'].shift(-1)
        nav_dataset['Strategy_Returns'] = np.where(nav_dataset['Predict'] == True  , nav_dataset['Next_Returns']  , -nav_dataset['Next_Returns'] )
        nav_dataset['Cumulative_Returns'] = np.cumsum(nav_dataset['Strategy_Returns'])
        nav_dataset = nav_dataset.iloc[: , 5:].drop(columns=['y_Reg'])
        plt.figure(figsize=(12,8))
        plt.plot(nav_dataset['Cumulative_Returns'], color='k',  alpha=0.60 )
        st.write('Score:' , round((nav_dataset.Cumulative_Returns[-2]) , 4 ))
        st.pyplot()
        nav_dataset = nav_dataset.dropna()
        return nav_dataset
    
    @property 
    def  trade (self):
        while True:
            deribit = self.ex_api
            dataset = self.deep
            a =  deribit.fetch_balance({'currency': 'ETH'})['total']['ETH']
            p = deribit.fetch_ticker(self.pair_trade)['info']['index_price']
            c = self.start_capital 
            diff =  (a * p) - self.start_capital 
            if (dataset.Predict[-1] == True) & (diff < -1.00) :
                deribit.create_market_buy_order(self.pair_trade , abs(diff))
                st.write( dataset.Predict[-1] , 'Buy' , round(diff , 2), round(p , 2) , round(a , 3))

            elif (dataset.Predict[-1] == False) & (diff > 1.00) :
                deribit.create_market_sell_order(self.pair_trade , abs(diff))
                st.write( dataset.Predict[-1] , 'Sell' , round(diff , 2) , round(p , 2), round(a , 3))
            else:
                st.write( dataset.Predict[-1] , 'Wait' , round(diff , 2) , round(p , 2), round(a , 3))

            latest_iteration = st.empty()
            bar = st.progress(0)
            for i in range(100):
                latest_iteration.text(f'Progress {i+1}')
                bar.progress(i + 1)
                sleep(self.sleep)
#____________________________________________________________________________     

model =  Run_model()
st.sidebar.header('(2020, 7 , 4) \n')
selectbox = lambda x, y : st.sidebar.selectbox('input_{}'.format(x),
    ( y ,'ad', 'ao', 'atr', 'bop', 'cci', 'cg', 'cmf', 'cmo', 'coppock', 'decreasing', 
    'dema', 'dpo', 'efi', 'ema', 'eom', 'fisher', 'fwma', 'hl2', 'hlc3', 'hma', 'increasing', 
    'kama', 'kurtosis', 'linear_decay', 'linreg', 'log_return', 'mad', 'median', 'mfi', 
    'midpoint', 'midprice', 'mom', 'natr', 'nvi', 'obv', 'ohlc4', 'percent_return', 'pvi', 
    'pvol', 'pvt', 'pwma', 'qstick', 'quantile', 'rma', 'roc', 'rsi', 'sinwma', 'skew', 'slope', 
    'sma', 'stdev', 'swma', 'seed' , 't3', 'tema' ,'trima', 'trix', 'true_range', 'uo', 
    'variance', 'vwap', 'vwma', 'willr', 'wma', 'zlma', 'zscore'))

st.sidebar.text("_"*45)
model.input_1 = selectbox(1 ,'rsi')
model.input_2 = selectbox(2 ,'rsi')
# model.input_3 = selectbox(3 ,'rsi')

st.sidebar.text("_"*45)
model.length_1 = st.sidebar.slider('length_1' , 2 , 500 , 20)
model.length_2 = st.sidebar.slider('length_2' , 2 , 500 , 40)
# model.length_3 = st.sidebar.slider('length_3' , 2 , 500 , 60)

st.sidebar.text("_"*45)
model.W_111 = st.sidebar.number_input('W_111' , -10.0 , 10.0 , model.W_111)
model.W_112 = st.sidebar.number_input('W_112' , -10.0 , 10.0 , model.W_112)
model.W_113 = st.sidebar.number_input('W_113' , -10.0 , 10.0 , model.W_113)
model.W_121 = st.sidebar.number_input('W_121' , -10.0 , 10.0 , model.W_121)
model.W_122 = st.sidebar.number_input('W_122' , -10.0 , 10.0 , model.W_122)
model.W_123 = st.sidebar.number_input('W_123' , -10.0 , 10.0 , model.W_123)
model.W_211 = st.sidebar.number_input('W_211' , -10.0 , 10.0 , model.W_211)
model.W_212 = st.sidebar.number_input('W_212' , -10.0 , 10.0 , model.W_212)
model.W_221 = st.sidebar.number_input('W_221' , -10.0 , 10.0 , model.W_221)
model.W_222 = st.sidebar.number_input('W_222' , -10.0 , 10.0 , model.W_222)

st.sidebar.text("_"*45)
model.pair_data = st.sidebar.text_input('data' , "TOMO-PERP")
model.timeframe = st.sidebar.selectbox('timeframe',('1h' , '5m' , '15m' , '1h', '4h' ,'1d'))
model.start_test =  np.datetime64(st.sidebar.date_input('start_test', value= dt.datetime(2020, 7, 4, 0, 0)))

st.sidebar.text("_"*45)
pyplot = model.chart
pyplot = model.nav
st.write(pyplot.iloc[: , :])

st.text("")
st.write('\n\nhttps://github.com/firstnattapon/test-stream/edit/master/app.py')
