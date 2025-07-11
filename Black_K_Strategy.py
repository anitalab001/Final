import pandas as pd 
import numpy as np 
import dataframe_image as dfi
from datetime import datetime
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
import seaborn as sns
sns.set(font=['sans-serif'])
sns.set_style("darkgrid",{"font.sans-serif":['Microsoft JhengHei']})
from tqdm import tqdm
import statsmodels.api as sm
from collections import defaultdict
import os
from pathlib import Path
import warnings
warnings.simplefilter(action='ignore')

BASE_DIR = Path(r"C:\Users\A51857\Desktop\study\交接項目\BlackK")
DATA_DIR = BASE_DIR/"Data"
PIC_DIR = BASE_DIR/"Backtest_Picture"

def count_mdd(df_profit,window):
    '''
    Funtion to count MDD
    df_profit(DataFrame): which contain profit by certain frequency.
    window(int): the duration of MDD.
    '''
    Roll_Max = df_profit.rolling(window, min_periods=1).max()
    Daily_Drawdown = df_profit-Roll_Max
    return Daily_Drawdown

def count_beta(marketdates,data,update_days=1):
    for d in marketdates[len(marketdates)-update_days:]:
        end_date = d
        dd = marketdates.index(d)
        start_date = marketdates[dd-19]
        stock_pool = data.loc[d]['股票代號'].to_list()
        for stock in stock_pool:
            window = data.loc[(data['股票代號']==stock)&(data.index>=start_date)&(data.index<=end_date)][['指數變化率','個股變化率']].dropna()
            if len(window)>=20:
                Y = window['指數變化率']
                X = window['個股變化率']
                X = sm.add_constant(X)
                model = sm.OLS(Y,X).fit()
                b = model.params['個股變化率']
                data.loc[(data['股票代號']==stock)&(data.index==end_date),'Beta'] = b

    return data

def update_index_data():
    '''
    Update index data to index historical data
    '''
    ## Old data
    df_index = pd.read_csv(DATA_DIR/'TWindex.csv',index_col=0).sort_index(ascending=True)
    df_index.index = pd.to_datetime(df_index.index)
    ## Update data
    Update_index = pd.read_excel(DATA_DIR/"daily_blackK_Update.xlsx",sheet_name='加權指數',header=None,usecols="B:F").loc[5:]
    Update_index.columns = ['日期','開盤價', '最高價', '最低價', '收盤價']
    Update_index['日期'] = pd.to_datetime(Update_index['日期'])
    Update_index = Update_index.set_index('日期')
    for c in Update_index.columns:
        Update_index[c]=Update_index[c].astype(float)
    if Update_index.index[-1]!= df_index.index[-1]:
        df_index = pd.concat([df_index,Update_index],axis=0)
        df_index.to_csv(DATA_DIR/'TWindex.csv')
        print("Finsish update TWindex.csv")
    else:
        print("Already update TWindex.csv")
    df_index['指數報酬率'] = df_index['收盤價'].pct_change(5)
    df_index['指數變化率'] = df_index['收盤價'].pct_change(1)
    df_index['SMA'] = df_index['收盤價'].rolling(90,min_periods=1).mean()
    return df_index

def update_stock_data(df_index):
    '''
     Update stock daily data to historical data
    '''
    data = pd.read_csv(DATA_DIR/"stock_daily_beta.csv",index_col=0).sort_index(ascending=True)
    data.index = pd.to_datetime(data.index)
    data = data[data.index.isna()==False]
    Update_data = pd.read_excel(DATA_DIR/"daily_blackK_Update.xlsx",sheet_name='上市日資料',header=None,index_col=False).loc[5:]
    Update_data.columns = ['股票代號', '股票名稱', '日期','開盤價', '最高價', '最低價', '收盤價', '漲跌', '漲幅(%)', '振幅(%)','成交量', '成交筆數', '成交金額(千)', '均張', '成交量變動(%)',
                           '均張變動(%)', '股本(百萬)','總市值(億)', '市值比重(%)', '本益比', '股價淨值比', '本益比(近四季)', '週轉率(%)', '成交值比重(%)','漲跌停', '均價', '成交量(股)']
    Update_data['日期'] = pd.to_datetime(Update_data['日期'])
    for c in Update_data.columns:
        try:
                Update_data[c] = Update_data[c].replace('- -',np.nan).astype(float)
        except:
                pass
    Update_data = Update_data.set_index('日期')

    if data.index[-1]!=Update_data.index[-1]:
        data = pd.concat([data,Update_data],axis=0)
        data.to_csv(DATA_DIR/"stock_daily_beta.csv")
        print("Finsish update stock_daily_beta.csv")
    else:
        print("Already update stock_daily_beta.csv")

    data['指數報酬率'] = df_index['指數報酬率']
    data['指數變化率'] = df_index['指數變化率']
    data['指數開盤價'] = df_index['開盤價']
    data['指數收盤價'] = df_index['收盤價']
    data['SMA'] = df_index['SMA']
    data['均張變動(%)'] = data['均張變動(%)'].replace('- -',np.nan).astype(float)
    data['股價淨值比'] = data['股價淨值比'].replace('- -',np.nan).astype(float)
    data['個股報酬率'] = data.groupby('股票代號')['收盤價'].pct_change(5)
    data['個股變化率'] = data.groupby('股票代號')['收盤價'].pct_change(1)
    data['調整報酬率'] = data['個股報酬率'] - data['指數報酬率']
    data['調整變化率'] = data['個股變化率'] - data['指數變化率']
    marketdates = data.sort_index(ascending=True).groupby(data.index).first().index.to_list()
    data = count_beta(marketdates,data)

    return data

def get_stockfuture():
    data_future = pd.read_csv(DATA_DIR/"stockfuture_pid.csv")
    clean_pids = []
    for pid in data_future['標的代號']:
        if str(pid).isnumeric()==True:
            clean_pids.append(float(pid))
        else:
            clean_pids.append(str(pid))
    data_future['標的代號'] = clean_pids
    return data_future['標的代號'].to_list()

def pick_stock(data):
    data['週轉變化率'] = data.groupby('股票代號')['週轉率(%)'].pct_change()
    data['週轉變化率_Q3'] = data.groupby('日期')['週轉變化率'].quantile(0.70)
    data['change'] = data.groupby('日期')['股價淨值比'].quantile(0.50)
    data['Beta_Q2'] = data.groupby('日期')['均張變動(%)'].quantile(0.50)
    data['Daily'] = np.where(data['指數開盤價']>data['SMA'],1,-1)

    df_stock = data[((data['收盤價']>30)&(data['指數開盤價']>data['SMA'])&(data['週轉率(%)']>2)&(data['股價淨值比']>data['change'])&(data['調整報酬率']<-0.05)&(data['Beta']>data['Beta_Q2']))|
                    ((data['收盤價']>30)&(data['指數開盤價']<data['SMA'])&(data['週轉變化率']<data['週轉變化率_Q3'])&(data['週轉率(%)']>2)&(data['股價淨值比']>data['change'])&(data['調整報酬率']<-0.05)&(data['Beta']>-1))]

    return df_stock

def group_byDay(df):
    df_group = df.groupby(df.index)['股票代號'].apply(list).to_frame()
    return df_group

def pick_duplicate(df_group,stockfuture_list,days=10):
    result = []
    for i in range(days,len(df_group)):
        window_start = df_group.iloc[i-days].name
        window_end = df_group.iloc[i].name
        window_df = df_group[(df_group.index >= window_start) & (df_group.index <= window_end)]
        stock_counter = defaultdict(int)
        for codes in window_df['股票代號']:
            for code in codes:
                stock_counter[code] += 1

        if data.loc[window_end]['指數開盤價'].values[0]>data.loc[window_end]['SMA'].values[0]:
            unique_stocks = [code for code, count in stock_counter.items() if (count >=3) & (count <=5) & (code in df_group.loc[window_end]['股票代號']) & (code in stockfuture_list)] 
        
        else:
            unique_stocks = [code for code, count in stock_counter.items() if (count >=2) & (count <=4) & (code in df_group.loc[window_end]['股票代號']) & (code in stockfuture_list)]
        if len(unique_stocks)!=0:
            result.append({
                '日期': window_end,
                '股票代號': unique_stocks
            })
    df_group = pd.DataFrame(result).set_index('日期',drop=True)
    return df_group

def pick_range(df_group,start,end):
    df_range = df_group.loc[start:end]
    for d in df_range.index:
        df_range.loc[d,'當日張數'] = int(5000000/(data.loc[(data.index==d)&(data['股票代號'].isin(df_range.loc[d]['股票代號']))]['收盤價'].sum()*1000))
    return df_range

def arrange_trade(n,stoploss,stopprofit,amount,marketdates_range,data,df_range):
    df = pd.DataFrame()
    ii=0
    for today in df_range.index[:]:
        today_index = marketdates_range.index(today)
        if today<= marketdates_range[-n-1]:
            tommorw = marketdates_range[today_index+1]
            next_ndays = marketdates_range[today_index+n]
            group = df_range.loc[today]['股票代號']
            amount_per = amount/len(group)
            for stock in group:
                window = data[(data.index>=today)&(data.index<=next_ndays)&(data['股票代號']==stock)]
                buy_date = tommorw
                buy_OHLC = '開盤價'
                try:
                    buy_price = window.loc[buy_date][buy_OHLC]*(1+0.004425)
                    buy_price_yes = window.loc[today]['收盤價']
                except:
                    buy_date = window.iloc[1].name
                    buy_price = window.iloc[1][buy_OHLC]*(1+0.004425)
                num = int(amount_per/(buy_price_yes*1000))
                # num = int(amount_per/(buy_price*1000))
                # same_num = df_range.loc[today]['當日張數']
                for w in range(1,len(window)-1):
                    stopprofit_cond = (window.iloc[w]['最高價']>buy_price*(1+stopprofit)*(1+window.iloc[w-1]['指數變化率']))
                    stoploss_cond = (window.iloc[w]['最低價']<=buy_price*(1-stoploss))
                    out_cond = (window.iloc[w]['調整報酬率']>0.025)&(window.iloc[w]['週轉率(%)']<2)&(window.iloc[w]['收盤價']<window.iloc[w]['SMA'])
                    if (stopprofit_cond) or (stoploss_cond)or (out_cond):
                        if stopprofit_cond:
                            if window.iloc[w]['最高價']==window.iloc[w]['收盤價']:
                                sell_date = window.iloc[w+1].name
                                sell_OHLC = '開盤價'
                                sell_position = '停利1'
                                break
                            else:
                                sell_date = window.iloc[w].name
                                sell_OHLC = '收盤價'
                                sell_position = '停利2'
                                break
                        elif stoploss_cond:
                            sell_date = window.iloc[w].name
                            sell_OHLC = '收盤價'
                            sell_position = '停損'
                            break
                        else:
                            sell_date = window.iloc[w+1].name
                            sell_OHLC = '開盤價'
                            sell_position = '條件出場'
                            break
                    else:
                        sell_date = window.iloc[-1].name
                        sell_OHLC = '開盤價'
                        sell_position = '一般出場'
                df.loc[ii,'Stock'] = stock
                df.loc[ii,'Num'] = num
                # df.loc[ii,'Num'] = same_num
                df.loc[ii,'BuyDate'] = buy_date
                df.loc[ii,'BuyOHLC'] = buy_OHLC
                df.loc[ii,'SellDate'] = sell_date
                df.loc[ii,'BuyDate'] = buy_date
                df.loc[ii,'SellOHLC'] = sell_OHLC
                df.loc[ii,'SellPosition'] = sell_position
                ii+=1
        else:
            group = df_range.loc[today]['股票代號']
            amount_per = amount/len(group)
            today_index = marketdates_range.index(today)
            if today_index ==len(marketdates_range)-1:
                tommorw=today
            else:
                tommorw = marketdates_range[today_index+1]
            for stock in group:
                window = data[(data.index>=today)&(data.index<=marketdates_range[-1])&(data['股票代號']==stock)]
                buy_date = tommorw
                buy_OHLC = '開盤價'
                buy_price = window.loc[buy_date][buy_OHLC]*(1+0.004425)
                buy_price_yes = window.loc[today]['收盤價']
                num = int(amount_per/(buy_price_yes*1000))
                # same_num = df_range.loc[today]['當日張數']
                if buy_date!=marketdates_range[-1]:
                    for w in range(1,len(window)-1):
                        stopprofit_cond = (window.iloc[w]['最高價']>buy_price*(1+stopprofit)*(1+window.iloc[w-1]['指數變化率']))
                        stoploss_cond = (window.iloc[w]['最低價']<=buy_price*(1-stoploss))
                        out_cond = (window.iloc[w]['調整報酬率']>0.025)&(window.iloc[w]['週轉率(%)']<2)&(window.iloc[w]['收盤價']<window.iloc[w]['SMA'])
                        if (stopprofit_cond) or (stoploss_cond) or (out_cond):
                            if stopprofit_cond:
                                if window.iloc[w]['最高價']==window.iloc[w]['收盤價']:
                                    sell_date = window.iloc[w+1].name
                                    sell_OHLC = '開盤價'
                                    sell_position = '停利1'
                                    break
                                else:
                                    sell_date = window.iloc[w].name
                                    sell_OHLC = '收盤價'
                                    sell_position = '停利2'
                                    break
                            elif stoploss_cond:
                                sell_date = window.iloc[w].name
                                sell_OHLC = '收盤價'
                                sell_position = '停損'
                                break
                            else:
                                sell_date = window.iloc[w+1].name
                                sell_OHLC = '開盤價'
                                sell_position = '條件出場'
                                break
                        else:
                            sell_date = marketdates_range[-1]
                            sell_OHLC = '收盤價'
                            sell_position = '未平倉'
                else :
                    sell_date = marketdates_range[-1]
                    sell_OHLC = '收盤價'
                    sell_position = '未平倉'
                df.loc[ii,'Stock'] = stock
                df.loc[ii,'Num'] = num
                # df.loc[ii,'Num'] = same_num
                df.loc[ii,'BuyDate'] = buy_date
                df.loc[ii,'BuyOHLC'] = buy_OHLC
                df.loc[ii,'SellDate'] = sell_date
                df.loc[ii,'SellOHLC'] = sell_OHLC
                df.loc[ii,'SellPosition'] = sell_position
                ii+=1
    return df
            
def Backtest(df,data,stoploss,stopprofit,amount,marketdates):
    ''' 
    df: Dataframe which contain stocks, buy,sell details
    '''
    df_backtest = pd.DataFrame(index=marketdates)
    df_delta = pd.DataFrame(index=marketdates)
    df_backtest_bytrade = pd.DataFrame()
    for i in tqdm(range(len(df[:]))):
        row = df.iloc[i]
        stock = row['Stock']
        num = row['Num']
        buy_date = row['BuyDate']
        buy_OHLC = row['BuyOHLC']
        sell_date = row['SellDate']
        sell_OHLC = row['SellOHLC']
        sell_condition = row['SellPosition']
        window = data.loc[(data.index>=buy_date)&(data.index<=sell_date)&(data['股票代號']==stock)]
        buy_price = window.loc[buy_date][buy_OHLC]*(1+0.004425)
        window['profit'] = (window['收盤價']*(1-0.001425)-buy_price)*1000*num
        window['Delta'] = window['收盤價']*1000*num
        if sell_condition=='停利1':
            sell_price = window.loc[sell_date][sell_OHLC]*(1-0.001425)
        elif sell_condition=='停利2':
            sell_price = window.loc[sell_date][sell_OHLC]*(1+stopprofit)*(1-0.001425)
        elif sell_condition=='停損':
            sell_price = buy_price*(1-stoploss)*(1-0.001425)
        elif sell_condition=='條件出場':
            sell_price = window.loc[sell_date][sell_OHLC]*(1-0.001425)
        else:
            sell_price = window.loc[sell_date]['收盤價']*(1-0.001425)
        window.loc[sell_date,'profit'] = (sell_price - buy_price)*1000*num
        window.loc[sell_date,'Delta']  =  sell_price*1000*num
        df_backtest_bytrade.loc[len(df_backtest_bytrade),['BuyPrice','SellPrice','Num']] = [buy_price,sell_price,num]
        df_backtest[str(stock)+str(i)] = window[['profit']]
        df_delta[str(stock)+str(i)] = window[['Delta']]

    df_delta = df_delta.fillna(0)
    df_delta['Delta'] = df_delta.sum(axis=1)
    df_backtest = df_backtest.fillna(method='ffill')
    df_backtest = df_backtest.fillna(0)
    df_backtest['Profit'] = df_backtest.sum(axis=1)
    df_backtest['daily_profit'] = df_backtest['Profit'] - df_backtest['Profit'].shift(1)
    df_backtest['Mdd'] = count_mdd(df_backtest['Profit'],len(df_backtest))
    df_backtest_bytrade['Profit'] = (df_backtest_bytrade['SellPrice']-df_backtest_bytrade['BuyPrice'])*df_backtest_bytrade['Num']*100
    return df_backtest,df_delta,df_backtest_bytrade

def plot_backtest(df_index,df,df_backtest,df_backtest_bytrade):
    df_TW = (df_index[(df_index.index>=df_backtest.index[0])&(df_index.index<=pd.to_datetime(end))]['收盤價'].pct_change()+1).cumprod()
    df_month = df_range.groupby([pd.Grouper(freq='ME')]).count()
    df_year = df_range.groupby([pd.Grouper(freq='Y')]).count()
    df_month_return = pd.concat([df_backtest[['Profit']].groupby(pd.Grouper(freq='ME')).first(),df_backtest[['Profit']].groupby(pd.Grouper(freq='ME')).last(),df_month[['當日張數']]],axis=1)
    df_month_return.columns = ['Start','End','Times']
    df_month_return['Profit'] = round(df_month_return['End'] - df_month_return['Start'])
    df_month_return['Return'] = round((df_month_return['Profit']/(np.where(df_month_return['Times']!=0,df_month_return['Times'],1)*5000000))*100,2)
    stopprofit_times = len(df[(df['SellPosition']=='停利1')|(df['SellPosition']=='停利2')])
    stoploss_times = len(df[(df['SellPosition']=='停損')])
    condition_times = len(df[(df['SellPosition']=='條件出場')])
    try:
        Unevendate = df[df['SellPosition']=='未平倉']['BuyDate'].sort_values(ascending=True).to_list()[0]
    except:
        Unevendate = df_backtest.index[-1]
    total_trade = len(df)
    month_profit_q1,month_profit_q2,month_profit_q3 = df_month_return['Profit'].quantile(0.25),df_month_return['Profit'].quantile(0.5),df_month_return['Profit'].quantile(0.75)
    month_return_q1,month_return_q2,month_return_q3 = df_month_return['Return'].quantile(0.25),df_month_return['Return'].quantile(0.5),df_month_return['Return'].quantile(0.75)
    month_max_loss = df_month_return['Profit'].min()
    month_max_loss_pct = df_month_return[df_month_return['Profit']==month_max_loss]['Return'].values[0]
    month_max_loss_date = str(df_month_return[df_month_return['Profit']==month_max_loss].index[0])[:7]
    month_max_win = df_month_return['Profit'].max()
    month_max_win_pct = df_month_return[df_month_return['Profit']==month_max_win]['Return'].values[0]
    month_max_win_date = str(df_month_return[df_month_return['Profit']==month_max_win].index[0])[:7]
    mean_win_rate = round((len(df_backtest[df_backtest['daily_profit']>0])/len(df_backtest))*100,2)
    already_profit = round(df_backtest.loc[Unevendate]['Profit'],2)
    unready_profit = round(df_backtest.iloc[-1]['Profit'] - already_profit,2)
    mean_delta = df_delta['Delta'].mean()
    pl_ratio = abs(df_backtest[df_backtest['daily_profit']>0]['daily_profit'].mean()/df_backtest[df_backtest['daily_profit']<0]['daily_profit'].mean())
    winrate_bytrade = len(df_backtest_bytrade[df_backtest_bytrade['Profit']>0])/len(df_backtest_bytrade)
    pl_ratio_bytrade = df_backtest_bytrade[df_backtest_bytrade['Profit']>0]['Profit'].mean()/max(1,abs(df_backtest_bytrade[df_backtest_bytrade['Profit']<0]['Profit'].mean()))
    exp_bytrade = pl_ratio_bytrade*winrate_bytrade - (1-winrate_bytrade)
    now_profit = df_backtest.iloc[-1]['Profit'] - df_backtest.loc['2025-06-09','Profit']

    ## Backtest numeric value
    df_backtest_result = pd.DataFrame()
    df_backtest_result.loc['停損','Result'] = f'{stoploss*100:.0f} % '
    df_backtest_result.loc['停利','Result'] = f'{stopprofit*100:.0f} % '
    df_backtest_result.loc['回測交易次數','Result'] = f'{total_trade:,.0f}'
    df_backtest_result.loc['回測停損/停利次數','Result'] = f'{stoploss_times:,.0f}/{stopprofit_times:,.0f}'
    df_backtest_result.loc['回測出場條件次數','Result'] = f'{condition_times:,.0f}'
    df_backtest_result.loc['日勝率','Result'] = f'{mean_win_rate:.2f} %'
    df_backtest_result.loc['交易勝率','Result'] = f'{winrate_bytrade*100:.2f} %'
    df_backtest_result.loc['月獲利Q1','Result'] = f'{month_profit_q1:,.0f}元 ({month_return_q1:.2f} %)'
    df_backtest_result.loc['月獲利Q2','Result'] = f'{month_profit_q2:,.0f}元 ({month_return_q2:.2f} %)'
    df_backtest_result.loc['月獲利Q3','Result'] = f'{month_profit_q3:,.0f}元 ({month_return_q3:.2f} %)'
    df_backtest_result.loc['月最大獲利','Result'] = f'{month_max_win:,.0f} ({month_max_win_pct:.2f} %), Month : {month_max_win_date}'
    df_backtest_result.loc['月最大損失','Result'] = f'{month_max_loss:,.0f} ({month_max_loss_pct:.2f} %), Month : {month_max_loss_date}'
    df_backtest_result.loc['日賺賠比','Result'] = f'{pl_ratio:.2f}'
    df_backtest_result.loc['交易賺賠比','Result'] = f'{pl_ratio_bytrade:.2f}'
    df_backtest_result.loc['交易期望值','Result'] = f'{exp_bytrade:.2f}'
    df_backtest_result.loc['日平均 Delta','Result'] = f'{mean_delta:,.0f}'
    df_backtest_result.loc['已實現損益','Result'] = f'{already_profit:,.0f}'
    df_backtest_result.loc['未實現損益','Result'] = f'{unready_profit:,.0f}'
    df_backtest_result.loc['策略實施後累積損益','Result'] = f'{now_profit:,.0f}'
    df_backtest_result.loc['未平倉日期','Result'] = f'{Unevendate}'
    dfi.export(df_backtest_result,PIC_DIR/"Backtest_Result.jpg")
    # print(f'Stop Loss : {stoploss*100} %')
    # print(f'Stop Profit : {stopprofit*100} %')
    # print(f'Total Trade : {total_trade}')
    # print(f'Mean win rate : {mean_win_rate} % ')
    # print(f'Monthly mean Profit : {month_profit_q1,month_profit_q2,month_profit_q3} 元')
    # print(f'Monthly mean Return : {month_return_q1,month_return_q2,month_return_q3} %')
    # print(f'Monthly max loss : {month_max_loss:,.0f} ({month_max_loss_pct:.2f}) %, Month : {month_max_loss_date}')
    # print(f'Monthly max profit : {month_max_win:,.0f} ({month_max_win_pct:.2f}) % , Month : {month_max_win_date}')
    # print(f'PL Ratio : '+str(abs(round(df_backtest['Profit'].mean()/df_backtest['Mdd'].mean(),4))))
    # print(f'Expect : {exp}')
    # print(f'Times of Stoploss : {stoploss_times} , Times of Stopprofit : {stopprofit_times}, Times of Condition Out : {condition_times}')
    # print(f'日平均 Delta : {mean_delta:,.0f}')
    # print(f'Uneven: {Unevendate}')
    # print(f'以實現損益: {already_profit:,.0f}')
    # print(f'未實現損益: {unready_profit:,.0f}')

    fig,ax1 = plt.subplots(3,figsize=(20,10),sharex=True,gridspec_kw={'height_ratios': [2,1,1]})
    fig.subplots_adjust(hspace = 0.005)
    ax1[0].set_title(f'Period : {start} ~ {end}',loc='center')
    ax2 = ax1[0].twinx()
    ax3 = ax2.twinx()
    ax2.grid(True,linestyle='--',alpha=0.5)
    ax3.grid(True,linestyle='--',alpha=0.5)
    ax1[0].get_yaxis().set_visible(False)
    color_map = {1: 'C02',-1: 'C08'}
    ax1[0].bar(df_backtest.index,100,alpha=0.2,width=5,linewidth=1,color=[color_map[i] for i in df_backtest['Daily']])
    ax2.axvline(Unevendate,color='red',linestyle='--',label='最早未平倉日期')
    ax2.axvline(pd.to_datetime('2025-06-09'),color='black',linestyle='--',alpha=0.5,label='實際執行日期')
    ax2.plot(df_TW,label='台灣加權指數',color="C10")
    ax3.plot(df_backtest['Profit'],color='C01',label='Cum Profit (元)',linewidth=2)
    ax2.scatter(df_TW[df_TW==df_TW.max()].index,df_TW.max(),color='C10',alpha=0.8)
    ax3.scatter(df_backtest[df_backtest['Profit']==df_backtest['Profit'].max()].index[-1],df_backtest[df_backtest['Profit']==df_backtest['Profit'].max()].iloc[-1]['Profit'],color='C01',alpha=0.8)
    stoplossdates = df[(df['SellPosition']=='停損')]['BuyDate']
    stopprofitdates = df[(df['SellPosition']=='停利1')|(df['SellPosition']=='停利2')]['BuyDate']
    ax3.scatter(stoplossdates,df_backtest.loc[stoplossdates]['Profit'],color='Red',alpha=0.5,s=20)
    ax3.scatter(stopprofitdates,df_backtest.loc[stopprofitdates]['Profit'],color='Green',alpha=0.5,s=20)
    ax2.text(df_TW[df_TW==df_TW.max()].index-pd.Timedelta(days=20),df_TW.max(),str(df_TW[df_TW==df_TW.max()].index.values[0])[5:10],color='C10',fontsize=10)
    ax3.text(df_backtest[df_backtest['Profit']==df_backtest['Profit'].max()].index[-1]-pd.Timedelta(days=20),df_backtest[df_backtest['Profit']==df_backtest['Profit'].max()].iloc[-1]['Profit'],str(df_backtest[df_backtest['Profit']==df_backtest['Profit'].max()].index.values[0])[5:10],color='C01',fontsize=10)
    ax3.tick_params(axis='y', colors='C01')
    ax2.tick_params(axis='y', colors='C10')

    lines, labels = ax1[0].get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    lines3, labels3 = ax3.get_legend_handles_labels()
    ax1[0].legend(lines + lines2 + lines3, labels + labels2 + labels3 , loc='upper left',facecolor='white',framealpha=1)

    ax1[1].plot(df_backtest.index,df_backtest['Mdd'],label='MDD',color='C01')
    ax1[1].fill_between(df_backtest.index,df_backtest['Mdd'],color='C01')
    ax1[1].set_ylim(top=0)
    ax1[2].plot(df_delta.index,-df_delta['Delta'],label='Delta',color='C03')
    ax1[2].fill_between(df_delta.index,-df_delta['Delta'],color='C03')
    ax1[2].set_ylim(top=0)
    try:
        ax1[1].scatter(df_backtest[df_backtest['Mdd']==df_backtest['Mdd'].min()].index[-1],df_backtest['Mdd'].min(),color='C10')
    except:
        pass
    ax1[1].text(df_backtest[df_backtest['Mdd']==df_backtest['Mdd'].min()].index[-1]-pd.Timedelta(days=20),df_backtest['Mdd'].min(),str(df_backtest[df_backtest['Mdd']==df_backtest['Mdd'].min()].index.values[0])[5:10]+'\n',color='Black',fontsize=10)
    ax1[1].legend(loc='upper left',facecolor='white',framealpha=1)
    ax1[2].scatter(df_delta[df_delta['Delta']==df_delta['Delta'].max()].index[-1],-df_delta['Delta'].max(),color='C10')
    ax1[2].text(df_delta[df_delta['Delta']==df_delta['Delta'].max()].index[-1]-pd.Timedelta(days=20),-df_delta['Delta'].max(),str(df_delta[df_delta['Delta']==df_delta['Delta'].max()].index.values[0])[5:10]+f" : {(df_delta['Delta'].max()):,.0f}\n ",color='Black',fontsize=10)
    ax1[2].legend(loc='upper left',facecolor='white',framealpha=1)
    plt.savefig(PIC_DIR/"Backtest.jpg")

def plot_price_distribution(df,data):
    df_price = pd.DataFrame()
    for i in range(len(df)):
        row = df.iloc[i]
        stock = row['Stock']
        buydate = row['BuyDate']
        buyprice = data.loc[(data['股票代號']==stock)&(data.index==buydate)]['開盤價'].values[0]
        df_price.loc[len(df_price),'Price']=buyprice
    df_price_group = df_price.groupby(pd.cut(df_price['Price'],np.arange(0,2500,50))).count()
    plt.figure(figsize=(10,5))
    plt.title(f'事件價錢分佈\nSample : {len(df_price)}')
    plt.xlabel('Price')
    plt.ylabel('Number')
    plt.grid(True,alpha=0.5,linestyle = '--')
    plt.bar(np.arange(0,2450,50),df_price_group['Price'],width=50)
    plt.savefig(PIC_DIR/"Price_Distribution.jpg")

def Notice_Board(df,data):
    # 盤中觀察名單
    df_notice = df[df['SellPosition']=='未平倉']
    stoplossprices = []
    stopprofitprices = []
    outs = []
    normal_outs = []
    for i in range(len(df_notice)):
        row = df_notice.iloc[i]
        stock = row['Stock']
        buydate = row['BuyDate']
        buyOHLC = row['BuyOHLC']
        buyprice = data.loc[(data['股票代號']==stock)&(data.index==buydate)][buyOHLC].values[0]*(1+0.004425)
        stoplossprice = buyprice*(1-stoploss)
        stopprofitprice = buyprice*(1+stopprofit)
        stoplossprices.append(round(stoplossprice,2))
        stopprofitprices.append(round(stopprofitprice,2))
        stock_now = data.loc[(data['股票代號']==stock)&(data.index==marketdates[-1])]
        out_cond = (stock_now['調整報酬率']>0.025)&(stock_now['週轉率(%)']<2)&(stock_now['收盤價']<stock_now['SMA'])
        if out_cond.values[0]==True:
            outs.append(True)
        else:
            outs.append(False)

        if marketdates.index(buydate)+19 == len(marketdates):
            normal_outs.append(len(marketdates)-marketdates.index(buydate))
        else:
            normal_outs.append(len(marketdates)-marketdates.index(buydate))
        
    df_notice['StopLoss'] = stoplossprices
    df_notice['StopProfit'] = stopprofitprices
    df_notice['Out'] = outs
    df_notice['NormalOut'] = normal_outs
    df_notice = df_notice.reset_index(drop=True)
    dfi.export(df_notice,PIC_DIR/"Notice.jpg")
    return df_notice

def action_for_tomorrow(df_range,df_notice):
    buy_dict = dict()
    sell_dict = dict()
    if df_range.index[-1]==marketdates[-1]:
        buystocks = df_range.loc[marketdates[-1]]['股票代號']
        per_amount = amount/len(buystocks)
        for stock in buystocks:
            buy_dict[stock] = int(per_amount/(data.loc[(data['股票代號']==stock)&(data.index==marketdates[-1])]['收盤價'].values[0]*1000))
        print(f'明日(下一交易日) 需買入股票 : {buy_dict}')
    else:
        print('明日(下一交易日) 無需買入股票')
    for i in range(len(df_notice)):
        row = df_notice.iloc[i]
        stock = row['Stock']
        num = row['Num']
        out = row['Out']
        normal_out = row['NormalOut']
        if normal_out==19 or out==True:
            sell_dict[stock] = num
    if len(sell_dict)!=0:
        print(f'明日(下一交易日) 開盤需賣出股票 : {sell_dict}')
    else:
        print(f'明日(下一交易日) 無需賣出股票')

def update_missing_data(update_days=2):
    ## Index
    df_index = pd.read_csv(DATA_DIR/'TWindex.csv',index_col=0).sort_index(ascending=True)
    df_index.index = pd.to_datetime(df_index.index)
    Update_index = pd.read_csv(DATA_DIR/"Missing_index_data.csv")
    Update_index['日期'] = pd.to_datetime(Update_index['日期'])
    Update_index = Update_index.set_index('日期')
    for c in Update_index.columns:
        Update_index[c]=Update_index[c].astype(float)
    if Update_index.index[-1]!= df_index.index[-1]:
        df_index = pd.concat([df_index,Update_index],axis=0)
        df_index.to_csv(DATA_DIR/'TWindex.csv')
        print("Finsish update TWindex.csv")
    df_index['指數報酬率'] = df_index['收盤價'].pct_change(5)
    df_index['指數變化率'] = df_index['收盤價'].pct_change(1)
    df_index['SMA'] = df_index['收盤價'].rolling(90,min_periods=1).mean()

    ## Stock
    data = pd.read_csv(DATA_DIR/"stock_daily_beta.csv",index_col=0).sort_index(ascending=True)
    data.index = pd.to_datetime(data.index)
    data = data[data.index.isna()==False]
    Update_data = pd.read_csv(DATA_DIR/"Missing_stock_data.csv")
    for c in Update_data.columns:
        try:
                Update_data[c] = Update_data[c].replace('- -',np.nan).astype(float)
        except:
                pass
    Update_data = Update_data.set_index('日期')

    if data.index[-1]!=Update_data.index[-1]:
        data = pd.concat([data,Update_data],axis=0)
        data.to_csv(DATA_DIR/"stock_daily_beta.csv")
        print("Finsish update stock_daily_beta.csv")

    data['指數報酬率'] = df_index['指數報酬率']
    data['指數變化率'] = df_index['指數變化率']
    data['指數開盤價'] = df_index['開盤價']
    data['指數收盤價'] = df_index['收盤價']
    data['SMA'] = df_index['SMA']
    data['均張變動(%)'] = data['均張變動(%)'].replace('- -',np.nan).astype(float)
    data['股價淨值比'] = data['股價淨值比'].replace('- -',np.nan).astype(float)
    data['個股報酬率'] = data.groupby('股票代號')['收盤價'].pct_change(5)
    data['個股變化率'] = data.groupby('股票代號')['收盤價'].pct_change(1)
    data['調整報酬率'] = data['個股報酬率'] - data['指數報酬率']
    data['調整變化率'] = data['個股變化率'] - data['指數變化率']
    marketdates = data.sort_index(ascending=True).groupby(data.index).first().index.to_list()
    data = count_beta(marketdates,data,update_days)

    return data,df_index

if __name__ == '__main__':
    ## 參數設定
    n = 20           #持有天數
    stoploss=0.15    #停損比例(15%)
    stopprofit=0.20  #停利比例(20%)
    amount = 5000000 #每天投入金額

    df_index = update_index_data()
    data = update_stock_data(df_index)

    # data,df_index = update_missing_data(update_days=2) # update+days 請填上缺失的天數

    marketdates = data.sort_index(ascending=True).groupby(data.index).first().index.to_list()

    start = '2020-01-01'  # 回測起始日
    end = marketdates[-1] # 回測截止日(今日)

    if end != datetime.today().replace(hour=0,minute=0,second=0,microsecond=0):
        print(f'今日尚未更新資料 (目前最新資訊為 : {end})')
    stockfuture_list = get_stockfuture()
    df_stock = pick_stock(data)
    df_group = pick_duplicate(group_byDay(df_stock),stockfuture_list,days=10)
    marketdates_range = [i for i in marketdates if (i>=pd.to_datetime(start))&(i<=pd.to_datetime(end))]
    df_range = pick_range(df_group,start,end)
    df_result = arrange_trade(n,stoploss,stopprofit,amount,marketdates_range,data,df_range)
    df_backtest,df_delta,df_backtest_bytrade = Backtest(df_result,data,stoploss,stopprofit,amount,marketdates_range)
    df_backtest['Daily'] = data['Daily'].groupby('日期').first()
    plot_backtest(df_index,df_result,df_backtest,df_backtest_bytrade)
    plot_price_distribution(df_result,data)
    df_notice = Notice_Board(df_result,data)
    action_for_tomorrow(df_range,df_notice)
    













