import logging


def log_init(path, name, mode, verbose=True):
    """Inits file writing logger object with INFO level

    Args:
        path (str): log path
        name (str): log file name
        mode (str): file writing mode
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

        file_handler = logging.FileHandler(f"{path}/{name}.txt", mode=mode)
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)

        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)

        logger.addHandler(file_handler)
        logger.addHandler(stream_handler)
        
    return logger