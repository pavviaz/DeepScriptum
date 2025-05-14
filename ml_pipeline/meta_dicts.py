from torch.optim import Adam, AdamW, SGD
import torch.nn.functional as F

from ml_pipeline.models import (
    VEDSingleModel,
    VEDDuoModel,
)
from ml_pipeline.datamodules import (
    ArxivOnePageDataModule,
)
from ml_pipeline.preprocessors import (
    arxiv_onepage,
    arxiv_onepage_processed,
)


def distill_loss(student_preds, gts, **kwargs):
    student_log_probs = F.log_softmax(student_preds / kwargs["temperature"], dim=-1)
    teacher_probs = F.softmax(kwargs["teacher_outputs"] / kwargs["temperature"], dim=-1)

    kl_loss = F.kl_div(student_log_probs, teacher_probs, reduction="batchmean") * (
        kwargs["temperature"] ** 2
    )

    ce_loss = F.cross_entropy(student_preds, gts)

    return kwargs["alpha"] * kl_loss + (1.0 - kwargs["alpha"]) * ce_loss


PREPROCESSORS = {
    "arxiv_onepage": arxiv_onepage,
    "arxiv_onepage_processed": arxiv_onepage_processed,
}

DATAMODULES = {
    "arxiv_onepage_ved": ArxivOnePageDataModule,
}

MODELS = {
    "VEDSingle": {
        "cls": VEDSingleModel,
        "dms": ["arxiv_onepage_ved"],
    },
    "VEDDuo": {
        "cls": VEDDuoModel,
        "dms": ["arxiv_onepage_ved"],
    },
}

LOSS_FUNCS = {
    "crossentropy": F.cross_entropy,
    "distill": distill_loss,
}

OPTIMS = {"adam": Adam, "adamw": AdamW, "sgd": SGD}
