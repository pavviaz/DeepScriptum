import os
import torch
from torch import nn
from munch import munchify


class Checkpointing:
    def __init__(self, path, max_to_keep=10, **kwargs) -> None:
        assert all(
            [
                (nn.Module in type(el).mro() or torch.optim.Optimizer in type(el).mro())
                for el in kwargs.values()
            ]
        )

        self.max_to_keep = max_to_keep
        self.modules = munchify(
            {k: v for k, v in kwargs.items() if nn.Module in type(v).mro()}
        )
        self.optims = munchify(
            {k: v for k, v in kwargs.items() if torch.optim.Optimizer in type(v).mro()}
        )

        if len(self.modules) == 0:
            print("Warning: no modules specified for saving/loading")
        if len(self.modules) == 0:
            print("Warning: no optimizers specified for saving/loading")

        self.path = path

        os.makedirs(os.path.dirname(self.path), exist_ok=True)

    def __get_files_and_ckpt_idx(self, return_all_checkpoints=False):
        dir_files = sorted(
            [
                s
                for s in os.listdir(self.path)
                if os.path.isfile(os.path.join(self.path, s))
            ],
            key=lambda s: os.path.getmtime(os.path.join(self.path, s)),
        )
        try:
            checkpoing_idx = (
                0
                if len(dir_files) == 0
                else (
                    (int(dir_files[-1].replace(".tar", "").split("_")[-1]) + 1)
                    if not return_all_checkpoints
                    else [
                        int(idx.replace(".tar", "").split("_")[-1]) + 1
                        for idx in dir_files
                    ]
                )
            )
        except:
            raise IOError
        return dir_files, checkpoing_idx

    def save(self):
        dir_files, checkpoing_idx = self.__get_files_and_ckpt_idx()

        del_idxs = len(dir_files) - self.max_to_keep + 1

        if del_idxs > 0:
            [
                os.remove(os.path.join(self.path, dir_files[idx]))
                for idx in range(del_idxs)
            ]

        try:
            torch.save(
                {k: v.state_dict() for k, v in self.modules.items()}
                | {k: v.state_dict() for k, v in self.optims.items()},
                os.path.join(self.path, f"checkpoint_{checkpoing_idx}.tar"),
            )
        except Exception as e:
            raise e

    def load(self, idx=None, print_avail_ckpts=False, return_idx=False):
        _, checkpoing_idx = self.__get_files_and_ckpt_idx(return_all_checkpoints=True)

        if print_avail_ckpts:
            print("Following checkpoints are available:")
            [
                print(f"{idx + 1}) {ckpt - 1}", sep=", ", end=" ")
                for idx, ckpt in enumerate(checkpoing_idx)
            ]

        checkpoing_idx = checkpoing_idx[-1] if not idx else idx + 1

        try:
            checkpoint = torch.load(
                os.path.join(self.path, f"checkpoint_{checkpoing_idx - 1}.tar")
            )
            [
                v.load_state_dict(checkpoint[k], strict=False)
                for k, v in self.modules.items()
            ]
            [v.load_state_dict(checkpoint[k]) for k, v in self.optims.items()]
        except Exception as e:
            raise e

        if return_idx:
            return checkpoing_idx
