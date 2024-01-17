def compute_threads_work(length, download_threads):
    """Yields split data for threads
    Args:
       length: number of captions in dataset
       download_threads: number of downloading threads
    Returns:
        (from, to) tuple
    """
    div, mod = divmod(length, download_threads)
    for _ in range(download_threads):
        yield (t := length - (div + bool(mod)), length)
        length = t
        mod -= 1 if mod else 0
