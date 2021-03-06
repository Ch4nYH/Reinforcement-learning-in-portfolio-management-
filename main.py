# -*- coding: utf-8 -*-
import json
from arguments import parser_args
from session import session
import random
import tensorflow as tf
import logging_config
import logging.config
import logging

logging.config.dictConfig(logging_config.logging_config)
logger = logging.getLogger()


def main():
    logger.info("Initializing")
    args = parser_args()
    tf.compat.v1.set_random_seed(args.seed)
    random.seed(args.seed)
    with open(args.config) as f:
        config = json.load(f)
        if args.mode == 'download':
            from data.download_data import DataDownloader
            data_downloader = DataDownloader(config)
            data_downloader.save_data()
        else:
            session(config, args)


if __name__ == "__main__":
    main()
