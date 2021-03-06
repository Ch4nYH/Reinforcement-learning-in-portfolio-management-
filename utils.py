import math
import numpy as np
from trader import StockTrader
import matplotlib.pyplot as plt
from agents.UCRP import UCRP
from agents.Loser import Loser
from agents.Winner import Winner
import logging
logger = logging.getLogger()


def parse_info(info):
    return info['reward'], info['continue'], info['next state'], info['weight vector'], info['price'], info['risk']


def traversal(stocktrader, agent, env, epoch, noise_flag, framework, method, trainable):
    info = env.step(None, None, noise_flag)
    r, done, state, w1, p, risk = parse_info(info)
    done = 1
    t = 0

    while done:
        w2 = agent.predict(state, w1)
        env_info = env.step(w1, w2, noise_flag)
        r, done, s_next, w1, p, risk = parse_info(env_info)
        if framework == 'PG':
            agent.save_transition(state, p, w2, w1)
        else:
            agent.save_transition(state, w2, r-risk, done, s_next, w1)
        loss, q_value, actor_loss = 0, 0, 0

        if framework == 'DDPG':
            if not done and trainable:
                agent_info = agent.train(method, epoch)
                loss, q_value = agent_info["critic_loss"], agent_info["q_value"]
                if method == 'model_based':
                    actor_loss = agent_info["actor_loss"]

        elif framework == 'PPO':
            if not done and trainable:
                agent_info = agent.train(method, epoch)
                loss, q_value = agent_info["critic_loss"], agent_info["q_value"]
                if method == 'model_based':
                    actor_loss = agent_info["actor_loss"]

        elif framework == 'PG':
            if not done and trainable:
                agent.train()

        stocktrader.update_summary(loss, r, q_value, actor_loss, w2, p)
        t = t + 1


def backtest(agent, env, path, framework):
    logger.debug("Backtest")

    agents = []
    agents.extend(agent)
    agents.append(UCRP())
    agents.append(Loser())
    agents.append(Winner())
    labels = [framework, 'UCRP', "Loser", "Winner"]

    wealths_result = []
    rs_result = []
    for i, agent in enumerate(agents):
        stocktrader = StockTrader()
        info = env.step(None, None, False)
        r, done, s, w1, p, risk = parse_info(info)
        done = 1
        wealth = 10000000
        wealths = [wealth]
        rs = [1]
        while done:
            w2 = agent.predict(s, w1)
            env_info = env.step(w1, w2, False)
            r, done, s_next, w1, p, risk = parse_info(env_info)
            wealth = wealth * math.exp(r)
            rs.append(math.exp(r)-1)
            wealths.append(wealth)
            s = s_next
            stocktrader.update_summary(0, r, 0, 0, w2, p)

        stocktrader.write(map(lambda x: str(x), env.get_codes()), labels[i])
        logger.debug('Finished agents {}'.format(i))
        wealths_result.append(wealths)
        rs_result.append(rs)

    logger.info('资产名称 \t 平均日收益率 \t 夏普率 \t 最大回撤')
    plt.figure(figsize=(8, 6), dpi=100)
    for i in range(len(agents)):
        plt.plot(wealths_result[i], label=labels[i])
        mrr = float(np.mean(rs_result[i])*100)
        sharpe = float(np.mean(rs_result[i])/np.std(rs_result[i])*np.sqrt(252))
        maxdrawdown = float(max(1 - min(wealths_result[i]) / np.maximum.accumulate(wealths_result[i])))
        logger.info("%s \t %s \t %s \t %s", labels[i], round(mrr, 3), round(sharpe, 3), round(maxdrawdown, 3))
    plt.legend()
    plt.savefig(path + 'backtest.png')
    #plt.show()


def parse_config(config, mode):
    num_codes = config["num_codes"]
    start_date = config["start_date"]
    end_date = config["end_date"]
    features = config["features"]
    agent_config = config["agents"]
    market = config["mark_types"]
    noise_flag, record_flag, plot_flag = config["noise_flag"], config["record_flag"], config["plot_flag"]
    predictor, framework, window_length = agent_config
    reload_flag, trainable = config['reload_flag'], config['trainable']
    method = config['method']
    epochs = config['epochs']
    if mode == 'test':
        record_flag = True
        noise_flag = False
        plot_flag = True
        reload_flag = True
        trainable = False
        method = 'model_free'

    logger.info("Status:")
    logger.info("Date: %s - %s", start_date, end_date)
    logger.info('Features: %s', features)
    logger.info("Predictor: %s, Framework %s, Window Length: %s", predictor, framework, window_length)
    logger.info("Epochs: %d", epochs)
    logger.info("Trainable: %d", trainable)
    logger.info("Reloaded Model: %d", reload_flag)
    logger.info("Method: %s", method)
    logger.info("Noise_flag: %d", noise_flag)
    logger.info("Record_flag: %d", record_flag)
    logger.info("Plot_flag %d: ", plot_flag)

    return num_codes, start_date, end_date, features, agent_config, market, predictor, framework, window_length, \
        noise_flag, record_flag, plot_flag, reload_flag, trainable, method, epochs
