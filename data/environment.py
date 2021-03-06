# -*- coding: utf-8 -*-
import random
import numpy as np
import pandas as pd


def fill_zeros(x):
    return '0'*(6-len(x)) + x


class Environment:
    def __init__(self, seed=1):
        self.cost = 0.0025
        self.seed = seed

    def get_repo(self, start_date, end_date, codes_num, market):
        self.data = pd.read_csv('./data/stock_data.csv', index_col=0, parse_dates=True, dtype=object)
        self.data["code"] = self.data["code"].astype(str)
        random.seed(3)
        print(set(self.data["code"]))
        codes = random.sample(set(self.data["code"]), 5)

        # codes = self.data["code"][:codes_num]
        data2 = self.data.loc[self.data["code"].isin(codes)]

        date_set = set(data2.loc[data2['code'] == codes[0]].index)
        for code in codes:
            date_set = date_set.intersection((set(data2.loc[data2['code'] == code].index)))

        date_set = date_set.intersection(set(pd.date_range(start_date, end_date)))
        self.date_set = list(date_set)
        self.date_set.sort()
        
        train_start_time = self.date_set[0]
        train_end_time = self.date_set[int(len(self.date_set) / 5) * 4 - 1]
        test_start_time = self.date_set[int(len(self.date_set) / 5) * 4]
        test_end_time = self.date_set[-1]
        codes = list(codes)

        return train_start_time, train_end_time, test_start_time, test_end_time, codes

    def get_data(self, start_time, end_time, features, window_length, market, codes):
        self.codes = codes
        self.data = pd.read_csv(r'./data/stock_data.csv', index_col=0, parse_dates=True, dtype=object)
        self.data["code"] = self.data["code"].astype(str)
        self.data[features] = self.data[features].astype(float)
        self.data = self.data[start_time.strftime("%Y-%m-%d"):end_time.strftime("%Y-%m-%d")]
        data = self.data

        self.M = len(codes)+1
        self.N = len(features)
        self.L = int(window_length)
        self.date_set = pd.date_range(start_time, end_time)
        print(codes)
        asset_dict = dict()
        for asset in codes:
            asset_data = data[data["code"] == asset].reindex(self.date_set).sort_index()
            asset_data = asset_data.resample('D').mean()
            asset_data['close'] = asset_data['close'].fillna(method='pad')
            base_price = asset_data.iloc[-1]['close']
            asset_dict[str(asset)] = asset_data
            asset_dict[str(asset)]['close'] = asset_dict[str(asset)]['close'] / base_price

            if 'high' in features:
                asset_dict[str(asset)]['high'] = asset_dict[str(asset)]['high'] / base_price

            if 'low' in features:
                asset_dict[str(asset)]['low'] = asset_dict[str(asset)]['low'] / base_price

            if 'open' in features:
                asset_dict[str(asset)]['open'] = asset_dict[str(asset)]['open'] / base_price
            if 'volume' in features:
                asset_dict[str(asset)]['volume'] = np.log(asset_dict[str(asset)]['volume'])

            asset_data = asset_data.fillna(method='bfill', axis=1)
            asset_data = asset_data.fillna(method='ffill', axis=1)  # 根据收盘价填充其他值
            # ***********************open as preclose*******************#
            # asset_data=asset_data.dropna(axis=0,how='any')
            asset_dict[str(asset)] = asset_data

        # 开始生成tensor
        self.states = []
        self.price_history = []
        self.prices = []
        t = self.L + 1
        length = len(self.date_set)
        self.wealth = 1e8
        while t < length - 1:
            state_close = np.ones(self.L)
            if 'high' in features:
                state_high = np.ones(self.L)
            if 'open' in features:
                state_open = np.ones(self.L)
            if 'low' in features:
                state_low = np.ones(self.L)
            if 'volume' in features:
                state_volume = np.ones(self.L)
            r = np.ones(1)
            state = []
            prices = np.ones(1)
            for asset in codes:
                asset_data = asset_dict[str(asset)]
                state_close = np.vstack((state_close, asset_data.iloc[t - self.L - 1:t - 1]['close']))
                if 'high' in features:
                    state_high = np.vstack((state_high, asset_data.iloc[t - self.L - 1:t - 1]['high']))
                if 'low' in features:
                    state_low = np.vstack((state_low, asset_data.iloc[t - self.L - 1:t - 1]['low']))
                if 'open' in features:
                    state_open = np.vstack((state_open, asset_data.iloc[t - self.L - 1:t - 1]['open']))
                if 'volume' in features:
                    state_volume = np.vstack((state_volume, asset_data.iloc[t - self.L - 1:t - 1]['volume']))
                r = np.vstack((r, asset_data.iloc[t]['close'] / asset_data.iloc[t - 1]['close']))
                prices = np.vstack((prices, asset_data.iloc[t]['close']))
            state.append(state_close)
            if 'high' in features:
                state.append(state_high)
            if 'low' in features:
                state.append(state_low)
            if 'open' in features:
                state.append(state_open)
            if 'volume' in features:
                state.append(state_volume)
            state = np.stack(state, axis=1)
            state = state.reshape(1, self.M, self.L, self.N)
            self.states.append(state)
            self.price_history.append(r)
            self.prices.append(prices)
            t = t + 1
        self.reset()

    def step(self, w1, w2, noise):  # w1为过去的动作, w2为当前的动作
        if self.done:
            not_terminal = 1
            price = self.price_history[self.t]
            stock_price = self.prices[self.t]

            if noise:
                price = price + np.stack(np.random.normal(0, 0.002, (1, len(price))), axis=1)
            mu = self.cost * (np.abs(w2[0][1:] - w1[0][1:])).sum()

            risk = 0
            r = (np.dot(w2, price)[0] - mu)[0]
            self.wealth = self.wealth
            reward = np.log(r + 1e-10)

            w2 = w2 / (np.dot(w2, price) + 1e-10)
            self.t += 1
            if self.t == len(self.states):
                not_terminal = 0
                self.reset()

            price = np.squeeze(price)
            info = {'reward': reward, 'continue': not_terminal, 'next state': self.states[self.t],
                    'weight vector': w2, 'price': price, 'risk': risk}
            return info
        else:
            info = {'reward': 0, 'continue': 1, 'next state': self.states[self.t],
                    'weight vector': np.array([[1] + [0 for i in range(self.M - 1)]]),
                    'price': self.price_history[self.t], 'risk': 0}

            self.done = True
            return info

    def reset(self):
        self.t = self.L + 1
        self.done = False

    def get_codes(self):
        return self.codes
