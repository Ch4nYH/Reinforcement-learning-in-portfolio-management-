# -*- coding: utf-8 -*-
"""
Notes: it's for downloading data
"""
import tushare as ts
import pandas as pd
ts.set_token("9e06d1e62fb2a9d01b75d20992b5767ef19c1df43662adc99c80a434")


class DataDownloader:
    def __init__(self, config):
        start_date = config["start_date"]
        end_date = config["end_date"]
        print(start_date)
        print(end_date)
        pro = ts.pro_api()
        stock_list = list(pro.stock_basic(exchange='', list_status='L', fields='ts_code')['ts_code'])
        self.stock_data = []
        for stock in stock_list[:100]:
            self.stock_data.extend(pro.daily(ts_code=stock, start_date=start_date, end_date=end_date).iloc[:, [1, 0, 2, 5, 3, 4, -2]].values)

    def save_data(self):
        df = pd.DataFrame(self.stock_data, columns=['date', 'code', 'open', 'close', 'high', 'low', 'volume' ])
        df['code'] = df['code'].apply(lambda x: x[:-3])
        df = df.sort_values(['date', 'code'])
        df.to_csv(
            'data/stock_data.csv', index=False)