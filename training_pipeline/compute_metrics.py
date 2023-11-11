import torch
import torcheval.metrics as metrics
import torcheval.metrics.functional as F
from matplotlib import pyplot as plt

from utils import disable_plot_show, enable_plot_show


class LossCounter:
    def __init__(self) -> None:
        self.reset()

    def update(self, val):
        self.loss += val
        self.cnt += 1

    def compute(self):
        return self.loss / self.cnt

    def reset(self):
        self.loss = 0
        self.cnt = 0


class F1Last:
    def __init__(self, num_classes, device=None):
        self.f1_last = metrics.MulticlassF1Score(
            num_classes=num_classes, average=None, device=device
        )

    @staticmethod
    def functional(**kwargs):
        return F.multiclass_f1_score(average=None, **kwargs)[-1]

    def update(self, y_pred, y_true):
        self.f1_last.update(y_pred, y_true)

    def compute(self):
        return self.f1_last.compute()[-1]

    def reset(self):
        self.f1_last.reset()


class RecallLast:
    def __init__(self, num_classes, device=None):
        self.recall_last = metrics.MulticlassRecall(
            num_classes=num_classes, average=None, device=device
        )

    @staticmethod
    def functional(**kwargs):
        return F.multiclass_recall(average=None, **kwargs)[-1]

    def update(self, y_pred, y_true):
        self.recall_last.update(y_pred, y_true)

    def compute(self):
        return self.recall_last.compute()[-1]

    def reset(self):
        self.recall_last.reset()


class PrecisionLast:
    def __init__(self, num_classes, device=None):
        self.precision_last = metrics.MulticlassPrecision(
            num_classes=num_classes, average=None, device=device
        )
    
    @staticmethod
    def functional(**kwargs):
        return F.multiclass_precision(average=None, **kwargs)[-1]

    def update(self, y_pred, y_true):
        self.precision_last.update(y_pred, y_true)

    def compute(self):
        return self.precision_last.compute()[-1]

    def reset(self):
        self.precision_last.reset()


class MetricsTracker:
    def __init__(self, num_classes, labels=None):
        self.conf_mat = metrics.MulticlassConfusionMatrix(num_classes)

        self.micro_accuracy = metrics.MulticlassAccuracy(average="micro")
        self.macro_accuracy = metrics.MulticlassAccuracy(
            num_classes=num_classes, average="macro"
        )
        self.micro_f1 = metrics.MulticlassF1Score(average="micro")
        self.macro_recall = metrics.MulticlassRecall(
            num_classes=num_classes, average="macro"
        )
        self.macro_precision = metrics.MulticlassPrecision(
            num_classes=num_classes, average="macro"
        )
        self.macro_f1 = metrics.MulticlassF1Score(
            num_classes=num_classes, average="macro"
        )
        self.recall_last = RecallLast(num_classes=num_classes)
        self.precision_last = PrecisionLast(num_classes=num_classes)
        self.f1_last = F1Last(num_classes=num_classes)

        self.loss = LossCounter()
        self.metrics = {
            "confusion_matrix": {
                "cls": self.conf_mat,
                "func": F.multiclass_confusion_matrix,
                "params": {"num_classes": num_classes},
            },
            "micro_accuracy": {
                "cls": self.micro_accuracy,
                "func": F.multiclass_accuracy,
                "params": {"average": "micro"},
            },
            "macro_accuracy": {
                "cls": self.macro_accuracy,
                "func": F.multiclass_accuracy,
                "params": {"num_classes": num_classes, "average": "macro"},
            },
            "micro_f1": {
                "cls": self.micro_f1,
                "func": F.multiclass_f1_score,
                "params": {"average": "micro"},
            },
            "macro_recall": {
                "cls": self.macro_recall,
                "func": F.multiclass_recall,
                "params": {"num_classes": num_classes, "average": "macro"},
            },
            "macro_precision": {
                "cls": self.macro_precision,
                "func": F.multiclass_precision,
                "params": {"num_classes": num_classes, "average": "macro"},
            },
            "macro_f1": {
                "cls": self.macro_f1,
                "func": F.multiclass_f1_score,
                "params": {"num_classes": num_classes, "average": "macro"},
            },
            "recall_last": {
                "cls": self.recall_last,
                "func": RecallLast.functional,
                "params": {"num_classes": num_classes},
            },
            "precision_last": {
                "cls": self.precision_last,
                "func": PrecisionLast.functional,
                "params": {"num_classes": num_classes},
            },
            "f1_last": {
                "cls": self.f1_last,
                "func": F1Last.functional,
                "params": {"num_classes": num_classes},
            },
        }

        self.labels = labels

    def update(self, y_pred, y_true, loss_val):
        """
        Updates internal state of all metrics

        Args:
            y_pred (torch.Tensor): Predicted labels (argmax).
            y_true (torch.Tensor): Target labels (argmax).
        """
        self.cur_y = {"input": y_pred, "target": y_true}
        self.cur_loss = loss_val

        [m["cls"].update(y_pred, y_true) for m in self.metrics.values()]
        self.loss.update(loss_val)

    def reset(self):
        """
        Resets internal state of all metrics
        """
        [m["cls"].reset() for m in self.metrics.values()]
        self.loss.reset()

    def compute_all(self, name=None):
        """
        Computes and returns dictionary with all metrics if ```name``` is not specified,
        or only one value for specific metric otherwise

        Args:
            name (str, optional): Name of one metric. Defaults to None.
        """

        if name and name in self.metrics:
            return self.metrics[name].compute()

        return {"loss": self.loss.compute()} | {
            k: v["cls"].compute() for k, v in self.metrics.items()
        }

    def compute_current(self, name=None):
        if name and name in self.metrics:
            return self.metrics[name]["func"](
                **self.cur_y, **self.metrics[name]["params"]
            )

        return {"loss": self.cur_loss} | {
            k: v["func"](**self.cur_y, **v["params"]) for k, v in self.metrics.items()
        }

    def plot_cm(self):
        disable_plot_show()

        fig, ax = plt.subplots(figsize=(6, 6))
        m = self.conf_mat.confusion_matrix.numpy()

        ax.matshow(m, cmap=plt.cm.Blues, alpha=0.3)

        if self.labels:
            ax.set_xticks(range(len(self.labels)))
            ax.set_xticklabels(self.labels, fontsize=6)

            ax.set_yticks(range(len(self.labels)))
            ax.set_yticklabels(self.labels, fontsize=6)

        for i in range(m.shape[0]):
            for j in range(m.shape[1]):
                ax.text(
                    x=j,
                    y=i,
                    s=m[i, j],
                    va="center",
                    ha="center",
                    size="xx-large",
                )

        ax.set_xlabel("Predictions", fontsize=6)
        ax.set_ylabel("Actuals", fontsize=6)
        ax.set_title("Confusion Matrix", fontsize=6)

        enable_plot_show()

        return fig


if __name__ == "__main__":
    import torch

    cm = MetricsTracker(3, ["class1", "class2", "class3"])

    y_pred = torch.tensor([1, 1, 2, 1, 0, 2, 0, 1, 2, 1, 1, 1])
    y_true = torch.tensor([1, 1, 2, 1, 2, 1, 0, 0, 0, 2, 2, 2])

    cm.update(y_pred[:5], y_true[:5], 1)
    print(cm.compute_current())
    cm.update(y_pred[5:], y_true[5:], 2)
    print(cm.compute_current())

    r = cm.compute_all()
    print(r)

    # cm.reset()

    p = cm.plot_cm()
    # p.show()
    # plt.show()
