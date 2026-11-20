# %%

import os

import h5py
import hdf5plugin  # noqa: F401
import numpy as np
import torch
from torch.utils.data import Dataset

"""
Pour les images : 
- on divise chaque canal par 255 et on normalise en utilisant les stats de ImageNet
Pour les actions : 
- on calcule moyenne et std des actions sur les épisodes de train et on normalise
"""

# %%

class PushTDataset(Dataset):
    def __init__(
        self,
        path,
        sequence_length=4,
        frame_skip=5,
        max_episodes=None,
        sample_stride=1,
        seed=42,
        action_stats_episodes=None,
    ):
        self.path = str(path)
        self.sequence_length = sequence_length
        self.frame_skip = frame_skip
        self.file = None
        self.file_pid = None

        with h5py.File(self.path, "r") as f:
            episode_idx = np.asarray(f["episode_idx"][:]).flatten()
            step_idx = np.asarray(f["step_idx"][:]).flatten()

            # only the actions and indexes are loaded here never the images.
            all_actions = np.asarray(f["action"][:], dtype=np.float32)

        if action_stats_episodes is not None:
            mask = np.isin(episode_idx, action_stats_episodes)
            actions_for_stats = all_actions[mask]
        else:
            actions_for_stats = all_actions

        finite_actions = actions_for_stats[np.isfinite(actions_for_stats).all(axis=1)]
        if len(finite_actions) < 2:
            raise ValueError("Not enough finite actions for normalization")

        self.action_mean = torch.from_numpy(finite_actions.mean(axis=0))
        self.action_std = torch.from_numpy(
            finite_actions.std(axis=0, ddof=1).clip(min=1e-6)
        )

        span = (sequence_length - 1) * frame_skip
        candidates = np.arange(max(0, len(episode_idx) - span))

        valid = (
            (episode_idx[candidates] == episode_idx[candidates + span])
            & (step_idx[candidates + span] - step_idx[candidates] == span)
        )
        starts = candidates[valid]

        if max_episodes is not None:
            episodes = np.unique(episode_idx)
            rng = np.random.default_rng(seed)
            selected = rng.choice(
                episodes, size=min(max_episodes, len(episodes)), replace=False
            )
            starts = starts[np.isin(episode_idx[starts], selected)]

        self.starts = starts[::sample_stride]
        if len(self.starts) == 0:
            raise ValueError("No valid PushT sequences found")

    def __len__(self):
        return len(self.starts)

    def _get_file(self):
        # open one HDF5 handle per DataLoader worker/process.
        pid = os.getpid()
        if self.file is None or self.file_pid != pid:
            if self.file is not None:
                self.file.close()
            self.file = h5py.File(self.path, "r")
            self.file_pid = pid
        return self.file

    def __getstate__(self):
        state = self.__dict__.copy()
        state["file"] = None
        state["file_pid"] = None
        return state

    def __getitem__(self, idx):
        f = self._get_file()
        start = int(self.starts[idx])
        T = self.sequence_length
        K = self.frame_skip

        frames = []
        for t in range(T):
            image = np.asarray(f["pixels"][start + t * K])
            # pixel between 0 and 1 
            image = torch.from_numpy(image.copy()).permute(2, 0, 1).float() / 255.0
            frames.append(image)

        observations = torch.stack(frames)  # (T, 3, 224, 224)

        mean = observations.new_tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = observations.new_tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        # normalization of observations using the statistics of ImageNet
        observations = (observations - mean) / std

        actions = np.asarray(f["action"][start : start + (T - 1) * K])
        actions = torch.from_numpy(actions.copy()).float()  # ((T-1)*K, 2)
        if not torch.isfinite(actions).all():
            raise ValueError("Found non-finite actions inside a training sequence")
        actions = (actions - self.action_mean) / self.action_std
        actions = actions.reshape(T - 1, 2 * K)

        return observations, actions

    def close(self):
        if self.file is not None:
            self.file.close()
            self.file = None
            self.file_pid = None