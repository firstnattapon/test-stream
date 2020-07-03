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
        self.pair_data = "BTC-PERP"
        self.pair_trade = 'BTC-PERPETUAL'
        self.apiKey ="AtdG0K3k"
        self.secret ="lItUXWckP2PNN-uPnrP_h_0dsctCXdFVP9x73bwo3Nc"
        self.Dense_11 = 0.03380605
        self.Dense_12 = -0.04777157
        self.Dense_21 = 0.00379837
        self.Dense_22 = 0.026092
        self.Dense_31 = -0.03202482
        self.Dense_32 = 0.04132303
        self.start_capital = 225.00
        self.sleep = 3
        self.timeframe = "1h"  
        self.limit = 500
        self.start_test = dt.datetime(2020, 7 , 3 , 0 , 0)
        self.length_1 = 21
        self.length_2 = 36
        
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
        dataset['input_1'] = dataset.ta.rsi(length= self.length_1 , scalar=1 , append=False)
        dataset['input_2'] = dataset.ta.rsi(length= self.length_2 , scalar=1 , append=False)
        dataset = dataset.fillna(0)
        dataset = dataset.dropna()
        dataset['y_Reg'] = dataset['OHLC4'].shift(-1).fillna(dataset.OHLC4[-1])
        X = dataset.iloc[ : , 1:-1]  ;  y_Reg = dataset.iloc[ : ,[ -1]] 
        return X , y_Reg , dataset
        
    @property  
    def deep (self):
        _,_, dataset = self.talib 
        dataset['Dense_1']  =  dataset.apply((lambda x :  max(0, ((self.Dense_11 * x.input_1)+(self.Dense_12  * x.input_2)+ 0))) , axis=1)
        dataset['Dense_2']  =  dataset.apply((lambda x :  max(0, ((self.Dense_21 * x.input_1)+(self.Dense_22  * x.input_2)+ 0))) , axis=1)
        dataset['Output']   =  dataset.apply((lambda x :  (((self.Dense_31) * x.Dense_1 ))+((self.Dense_32) * x.Dense_2 )+ 0 ) , axis=1)
        dataset['Predict']  =  dataset.Output.shift(1) <  dataset.Output.shift(0)
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
st.sidebar.header('(2020, 7 , 3) \n')

# model.pair_trade = st.sidebar.text_input('Symbol' , 'ETH-PERPETUAL')
# model.apiKey = st.sidebar.text_input('apiKey' , "AtdG0K3k")
# model.secret = st.sidebar.text_input('apiKey' ,"lItUXWckP2PNN-uPnrP_h_0dsctCXdFVP9x73bwo3Nc")
# model.start_capital = st.sidebar.slider('start_capital' , 0 , 500 , 225)
# model.sleep = st.sidebar.slider('sleep' , 0.0 , 6.0 , 3.0)

st.sidebar.text("_"*50)
st.sidebar.text("start_capital : {}".format (model.start_capital))
st.sidebar.text("Dense_11 : {}".format (model.Dense_11))
st.sidebar.text("Dense_12 : {}".format (model.Dense_12))
st.sidebar.text("Dense_21 : {}".format (model.Dense_21))
st.sidebar.text("Dense_22 : {}".format (model.Dense_22))
st.sidebar.text("Dense_31 : {}".format (model.Dense_31))
st.sidebar.text("Dense_32 : {}".format (model.Dense_32))
st.sidebar.text("_"*50)

model.length_1 = st.sidebar.slider('length_1' , 2 , 100 , 21)
model.length_2 = st.sidebar.slider('length_2' , 2 , 100 , 36)

# if st.sidebar.button('Run_model'):
#         model =  Run_model()
#         model.trade

model.pair_data = st.sidebar.text_input('data' , "BTC-PERP")
model.timeframe = st.sidebar.selectbox('timeframe',('1h' , '5m' , '15m' , '1h', '4h' ,'1d'))
model.start_test =  np.datetime64(st.sidebar.date_input('start_test', value= dt.datetime(2020, 7, 3, 0, 0)))
pyplot = model.chart
pyplot = model.nav
st.write(pyplot.iloc[: , 6:])
st.write('\n\nhttps://github.com/firstnattapon/test-stream/edit/master/app.py')
