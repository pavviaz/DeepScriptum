from clearml import Task, Logger
from datetime import datetime


class TrainingLogger:
    def __init__(self, clearml_logger: Logger) -> None:
        self.logger = clearml_logger

    def _add_timestamp(func):
        def wrapper(self, msg):
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            msg = f"[{timestamp}] {msg}"
            return func(self, msg)

        return wrapper

    @_add_timestamp
    def info(self, msg):
        self.logger.report_text(msg)

    def log_cm(self, **kwargs):
        self.info(f"Reported confusion matrix with params: {str(kwargs)}")
        self.logger.report_confusion_matrix(**kwargs, series="ignored")

    def log_scalar(self, **kwargs):
        self.logger.report_scalar(**kwargs)

    def invoke_exception(self, msg, exc, task: Task = None):
        self.info(msg)
        if task:
            task.mark_failed()

        raise exc(msg)
