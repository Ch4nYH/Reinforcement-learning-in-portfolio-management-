import os
import json
from agents.pg import PG
from agents.ddpg import DDPG
from trader import StockTrader
from datetime import datetime

from utils import backtest, traversal, parse_config
from data.environment import Environment

import logging
logger = logging.getLogger()


def session(config, args):
    codes, start_date, end_date, features, agent_config, \
        market, predictor, framework, window_length, noise_flag, record_flag, \
        plot_flag, reload_flag, trainable, method, epochs = parse_config(config, args)
    env = Environment(args.seed)

    stocktrader = StockTrader()
    path = "result/{}/{}/".format(framework, args.num)
    logger.info('Mode: {}'.format(args.mode))

    if args.mode == 'train':
        if not os.path.exists(path):
            os.makedirs(path)
            train_start_date, train_end_date, test_start_date, test_end_date, codes = env.get_repo(start_date, end_date, codes, market)
            logger.debug("Training with codes: {}".format(codes))
            env.get_data(train_start_date, train_end_date, features, window_length, market, codes)
            with open(path + 'config.json', 'w') as f:
                print(train_start_date)
                print(train_end_date)
                print(test_start_date)
                print(test_end_date)
                json.dump({"train_start_date": train_start_date.strftime('%Y-%m-%d'),
                           "train_end_date": train_end_date.strftime('%Y-%m-%d'),
                           "test_start_date": test_start_date.strftime('%Y-%m-%d'),
                           "test_end_date": test_end_date.strftime('%Y-%m-%d'), "codes": codes}, f)
        else:
            with open('result/{}/{}/config.json'.format(framework, args.num), 'r') as f:
                dict_data = json.load(f)
            train_start_date, train_end_date, codes = datetime.strptime(dict_data['train_start_date'], '%Y-%m-%d'), datetime.strptime(
                dict_data['train_end_date'], '%Y-%m-%d'), dict_data['codes']
            env.get_data(train_start_date, train_end_date, features, window_length, market, codes)

        if framework == 'PG':
            logger.debug("Loading PG Agent")
            agent = PG(len(codes) + 1, int(window_length), len(features), '-'.join(agent_config), reload_flag, trainable, args.num)
        elif framework == 'DDPG':
            logger.debug("Loading DDPG Agent")
            agent = DDPG(len(codes) + 1, int(window_length), len(features), '-'.join(agent_config), reload_flag, trainable, args.num)

        logger.info("Training: %d epochs", epochs)
        for epoch in range(epochs):
            traversal(stocktrader, agent, env, epoch, True, framework, method, trainable)

            if record_flag:
                stocktrader.write(epoch, framework)

            if plot_flag:
                stocktrader.plot_result()

            agent.reset_buffer()
            stocktrader.print_result(epoch, agent, True)
            stocktrader.reset()
        agent.close()

    elif args.mode == 'test':
        
        with open("result/{}/{}/config.json".format(framework, args.num), 'r') as f:
            dict_data = json.load(f)
        test_start_date, test_end_date, codes = datetime.strptime(dict_data['test_start_date'], '%Y-%m-%d'), datetime.strptime(dict_data['test_end_date'], '%Y-%m-%d'), dict_data['codes']
        env.get_data(test_start_date, test_end_date, features, window_length, market, codes)
        if framework == 'PG':
            logger.info("Loading PG Agent")
            agent = PG(len(codes) + 1, int(window_length), len(features), '-'.join(agent_config), True, False, args.num)
        elif framework == 'DDPG':
            logger.info("Loading DDPG Agent")
            agent = DDPG(len(codes) + 1, int(window_length), len(features), '-'.join(agent_config), True, False, args.num)
        backtest([agent], env, "result/{}/{}/".format(framework, args.num), framework)
