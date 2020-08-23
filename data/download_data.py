# -*- coding: utf-8 -*-
"""
Notes: it's for downloading data
"""
import tushare as ts
import pandas as pd
ts.set_token("9e06d1e62fb2a9d01b75d20992b5767ef19c1df43662adc99c80a434")
class DataDownloader:
    def __init__(self,config):
        start_date = config["data"]["start_date"]
        end_date = config["data"]["end_date"]
        market_types = config["data"]["mark_types"]
        pro = ts.pro_api()
        for market in market_types:
            if market=='stock':
                stock_list=list(pro.stock_basic(exchange='', list_status='L', fields='ts_code')['ts_code'])
                self.stock_data=[]
                for stock in stock_list:
                    self.stock_data.extend(pro.daily(ts_code = stock,start=start_date,end=end_date).iloc[:,[1,2,5,3,4,-2,0]].values)
    def save_data(self):
        pd.DataFrame(self.stock_data, columns=['time', 'open', 'close', 'high', 'low', 'volume', 'code']).to_csv(
            'stock_data.csv', index=False)