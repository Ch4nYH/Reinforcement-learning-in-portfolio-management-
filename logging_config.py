import logging

logging_config = dict(
    version=1,
    formatters={
        'simple': {'format': '%(asctime)s %(levelname)s {path: %(pathname)s} %(message)s'}
    },
    handlers={
        'default_handlers': {'class': 'logging.handlers.RotatingFileHandler',
                             'filename': './logfile/logger.log',
                             'level': 'INFO',
                             'formatter': 'simple',
                             'encoding': 'utf8'}
    },

    root={
        'handlers': ['default_handlers'],
        'level': logging.DEBUG,
    },
)