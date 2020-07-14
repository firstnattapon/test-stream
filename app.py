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
from cryptorandom.cryptorandom import SHA256
from scipy import special as s
from sympy import nextprime
# sns.set_style("whitegrid")

class Run_model :
    def __init__(self , ex='deribit'):
        self.ex = ex
        self.pair_data = "TOMO-PERP"
        self.pair_trade = 'TOMO-PERPETUAL'
        self.apiKey ="AtdG0K3k"
        self.secret ="lItUXWckP2PNN-uPnrP_h_0dsctCXdFVP9x73bwo3Nc"
        self.W_11 = -0.01505379
        self.W_12 = -0.00130575
        self.W_21 =  0.00302326
        self.W_22 =  0.02617892
        self.W_31 = -0.03784165
        self.W_32 =  0.00646918
        self.W_41 =  0.4325013 
        self.W_42 = -0.27094534
        self.W_43 =  1.0962713
        self.start_capital = 225.00
        self.sleep = 3
        self.timeframe = "1h"  
        self.limit = 500
        self.start_test = dt.datetime(2020, 7 , 4 , 0 , 0)
        self.length_1 = 50
        self.length_2 = 1
        self.input_1  = 'slope'
        self.input_2  = 'ad'
        
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
        
        if self.input_1 == 'jv':
            dataset['input_1'] = dataset.OHLC4.map(lambda x : s.jv(np.log(self.length_1) , x ))
        elif self.input_1 == 'nextprime':
            dataset['input_1'] = dataset.OHLC4.map(lambda x : nextprime(x*10 , self.length_1))
        else:
            dataset['input_1'] = dataset.ta(kind=self.input_1 , length= self.length_1 , scalar=1 , append=False)
            
        #______________
            
        if self.input_2 == 'jv':
            dataset['input_2'] = dataset.OHLC4.map(lambda x : s.jv(np.log(self.length_2) , x))
        elif self.input_2 == 'nextprime':
            dataset['input_2'] = dataset.OHLC4.map(lambda x : nextprime( x*10 , self.length_2))
        else:
            dataset['input_2'] = dataset.ta(kind=self.input_2 , length= self.length_2 , scalar=1 , append=False)   
            
        dataset = dataset.fillna(0)
        dataset['y_Reg'] = dataset['OHLC4'].shift(-1).fillna(dataset.OHLC4[-1])
        X = dataset.iloc[ : , 1:-1]  ;  y_Reg = dataset.iloc[ : ,[ -1]] 
        return X , y_Reg , dataset
        
    @property  
    def deep (self):
        _,_, dataset = self.talib 
        dataset['Dense_1']  = dataset.apply((lambda x :  max(0, ((self.W_11 * x.input_1)+(self.W_12  * x.input_2)+ -0.03362045))) , axis=1)
        dataset['Dense_2']  = dataset.apply((lambda x :  max(0, ((self.W_21 * x.input_1)+(self.W_22  * x.input_2)+  0.01533893))) , axis=1)
        dataset['Dense_3']  = dataset.apply((lambda x :  max(0, ((self.W_31 * x.input_1)+(self.W_32  * x.input_2)+ -0.03311799))) , axis=1)
        dataset['Output']   =  dataset.apply((lambda x : (((self.W_41) * x.Dense_1))+((self.W_42) * x.Dense_2)+((self.W_43)* x.Dense_3) + -0.04521269) , axis=1)
        dataset['Predict']  =  dataset.Output.shift(1) <  dataset.Output.shift(0)
        dataset = dataset.dropna()
        return dataset
    
    @property 
    def chart (self):
        dataset = self.deep
        dataset['buy'] =  dataset.apply(lambda x : np.where( x.Predict == True , x.OHLC4 , None) , axis=1)
        dataset['sell'] = dataset.apply(lambda x : np.where( x.Predict == False, x.OHLC4 , None) , axis=1)
        if st.checkbox('chart_plot'):
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
        nav_dataset['CumulativeMarket_Returns'] = np.cumsum(nav_dataset['Next_Returns'])
        #__________        
        nav_dataset['Strategy_Returns'] = np.where(nav_dataset['Predict'] == True  , nav_dataset['Next_Returns']  , -nav_dataset['Next_Returns'] )
        nav_dataset['Cumulative_Returns'] = np.cumsum(nav_dataset['Strategy_Returns'])
        #__________
        nav_dataset['Max_Returns'] = np.where(nav_dataset['Predict'] == True  , abs(nav_dataset['Next_Returns'])  , abs(-nav_dataset['Next_Returns']))
        nav_dataset['CumulativeMax_Returns'] = np.cumsum(nav_dataset['Max_Returns'])
        nav_dataset = nav_dataset.iloc[: , 5:].drop(columns=['y_Reg'])
        if st.checkbox('nav_plot'):
            plt.figure(figsize=(12,8))
            plt.plot(nav_dataset['Cumulative_Returns'], color='k',  alpha=0.60 )
            plt.plot(nav_dataset['CumulativeMax_Returns'], color='g',  alpha=0.60 )
            plt.plot(nav_dataset['CumulativeMarket_Returns'], color='r',  alpha=0.60 )
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
    'sma', 'stdev', 'swma', 'jv' , 't3', 'tema' ,'trima', 'trix', 'true_range', 'uo', 
    'variance', 'vwap', 'vwma', 'willr', 'wma', 'zlma', 'zscore' ,'nextprime'))

st.sidebar.text("_"*45)
model.input_1 = selectbox(1 ,'slope')
model.input_2 = selectbox(2 ,'ad')

st.sidebar.text("_"*45)
model.length_1 = st.sidebar.slider('length_1' , 1 , 500 , 50)
model.length_2 = st.sidebar.slider('length_2' , 1 , 500 , 1)

st.sidebar.text("_"*45)
model.W_11 = st.sidebar.number_input('W_11' , -10.0 , 10.0 , model.W_11)
model.W_12 = st.sidebar.number_input('W_12' , -10.0 , 10.0 , model.W_12)
model.W_21 = st.sidebar.number_input('W_21' , -10.0 , 10.0 , model.W_21)
model.W_22 = st.sidebar.number_input('W_22' , -10.0 , 10.0 , model.W_22)
model.W_31 = st.sidebar.number_input('W_31' , -10.0 , 10.0 , model.W_31)
model.W_32 = st.sidebar.number_input('W_32' , -10.0 , 10.0 , model.W_32)
model.W_41 = st.sidebar.number_input('W_41' , -10.0 , 10.0 , model.W_41)
model.W_42 = st.sidebar.number_input('W_42' , -10.0 , 10.0 , model.W_42)
model.W_43 = st.sidebar.number_input('W_43' , -10.0 , 10.0 , model.W_43)

st.sidebar.text("_"*45)
model.pair_data = st.sidebar.text_input('data' , "TOMO-PERP")
model.timeframe = st.sidebar.selectbox('timeframe',('1h' , '5m' , '15m' , '1h', '4h' ,'1d'))
model.start_test =  np.datetime64(st.sidebar.date_input('start_test', value= dt.datetime(2020, 7, 4, 0, 0)))

st.sidebar.text("_"*45)
pyplot = model.chart
pyplot = model.nav
if st.checkbox('df_plot'):
    st.write(pyplot.iloc[: , :])
st.text("")
st.write('\n\nhttps://github.com/firstnattapon/test-stream/edit/master/app.py')
                                
