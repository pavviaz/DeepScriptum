import logging
import collections
import os

import matplotlib


def log_init(stream, verbose=True):
    """Inits file writing logger object with INFO level

    Args:
        stream: 
        verbose (str): turns logger off if False

    Returns:
        Logger: logger object
    """
    logger = logging.getLogger(__name__)
    logger.handlers.clear()
    
    logger.setLevel(logging.INFO)
    logger.disabled = True

    if verbose:
        logger.disabled = False
        formatter = logging.Formatter("%(asctime)s:    %(message)s")

        file_handler = logging.StreamHandler(stream)
        file_handler.setFormatter(formatter)

        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)

        logger.addHandler(file_handler)
        logger.addHandler(stream_handler)
        
    return logger


def flatten_dict(dictionary, parent_key=False, separator='.'):
    """
    Turn a nested dictionary into a flattened dictionary
    :param dictionary: The dictionary to flatten
    :param parent_key: The string to prepend to dictionary's keys
    :param separator: The string used to separate flattened keys
    :return: A flattened dictionary
    """

    items = []
    for key, value in dictionary.items():
        new_key = str(parent_key) + separator + key if parent_key else key
        if isinstance(value, collections.abc.MutableMapping):
            items.extend(flatten_dict(value, new_key, separator).items())
        elif isinstance(value, list):
            for k, v in enumerate(value):
                items.extend(flatten_dict({str(k): v}, new_key).items())
        else:
            items.append((new_key, value))
    return dict(items)


def disable_plot_show():
    matplotlib.use('Agg')

    
def enable_plot_show():
    matplotlib.use(os.getenv("MATPLOTLIB_BACKEND"))