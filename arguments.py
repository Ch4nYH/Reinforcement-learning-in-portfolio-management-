from argparse import ArgumentParser


def parser_args():
    parser = ArgumentParser(description='Provide arguments for training different DDPG or PPO models in Portfolio Management')
    parser.add_argument("--mode", choices=['train', 'test', 'download'])
    parser.add_argument("--num", type=int)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--config", type=str, default='config.json')
    args = parser.parse_args()
    return args
