import os
from glob import glob
from typing import List, Dict, Any, Optional, Tuple


def preprocessor(
    path: str,
    log_obj: Any,
    task: Any,
):
    if not os.path.exists(path):
        error_msg = f"Data folder '{path}' doesn't exist"
        log_obj.invoke_exception(error_msg, OSError, task)
        return []

    datafiles = glob(f"{path}/*.json")
    if not len(datafiles):
        error_msg = f"No files in '{path}'"
        log_obj.invoke_exception(error_msg, OSError, task)
        return []

    return datafiles


if __name__ == "__main__":

    class MockLogger:
        def info(self, text):
            print(text)

        def invoke_exception(self, text, *args):
            print(text)

    paths = preprocessor(
        "processed_data",
        MockLogger(),
        None,
    )
    print(paths[:5])
