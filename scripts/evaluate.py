# %%

from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from leworldmodel.data.pusht import PushTDataset
from leworldmodel.model import LeWorldModel

# %%

project_root = next(
    p for p in [Path.cwd(), *Path.cwd().parents]
    if (p / "pyproject.toml").exists()
)

device = "cuda" if torch.cuda.is_available() else "cpu"

torch.manual_seed(3072)

B = 16
T = 4
K = 5

print("Device:", device)

# %%
# Load the trained model

# checkpoint_path = project_root / "checkpoints" / "run_01" / "best.pt"
checkpoint_path = project_root / "checkpoints" / "last.pt"

checkpoint = torch.load(
    checkpoint_path,
    map_location=device,
    weights_only=True,
)

model = LeWorldModel(
    params_encoder=checkpoint["params_encoder"],
    params_predictor=checkpoint["params_predictor"],
).to(device)

model.load_state_dict(checkpoint["model"])
model.eval()

print("Loaded epoch:", checkpoint["epoch"])
print("Training loss:", checkpoint.get("train_loss", checkpoint.get("loss")))

if "val_loss" in checkpoint:
    print("Validation loss:", checkpoint["val_loss"])

# %%
# Reconstruct the test episodes

data_path = project_root / "data" / "pusht" / "pusht_expert_train.h5"

with h5py.File(data_path, "r") as f:
    episode_idx = np.asarray(f["episode_idx"][:]).flatten()

full_dataset = PushTDataset(
    path=data_path,
    sequence_length=T,
    frame_skip=K,
    max_episodes=None,
    sample_stride=5,
    action_stats_episodes=checkpoint.get("train_episodes") if "action_mean" in checkpoint else None,
)

if "action_mean" in checkpoint:
    full_dataset.action_mean = checkpoint["action_mean"].cpu()
    full_dataset.action_std = checkpoint["action_std"].cpu()

if "test_episodes" in checkpoint:
    test_episodes = np.asarray(checkpoint["test_episodes"])
else:
    train_dataset = PushTDataset(
        path=data_path,
        sequence_length=T,
        frame_skip=K,
        max_episodes=1000,
        sample_stride=5,
        seed=42,
    )
    train_episodes = np.unique(episode_idx[train_dataset.starts])
    test_episodes = np.setdiff1d(np.unique(episode_idx), train_episodes)

all_sequence_episodes = episode_idx[full_dataset.starts]

test_indices = np.flatnonzero(
    np.isin(all_sequence_episodes, test_episodes)
)

test_dataset = Subset(full_dataset, test_indices.tolist())

assert len(test_dataset) > 0

if "train_episodes" in checkpoint:
    assert not np.isin(test_episodes, checkpoint["train_episodes"]).any()

if "val_episodes" in checkpoint:
    assert not np.isin(test_episodes, checkpoint["val_episodes"]).any()

print("Test episodes:", len(test_episodes))
print("Test sequences:", len(test_dataset))

# %%

test_dataloader = DataLoader(
    test_dataset,
    batch_size=B,
    shuffle=True,
    drop_last=False,
    num_workers=0,
    pin_memory=(device == "cuda"),
    generator=torch.Generator().manual_seed(3072),
)

# %%
# Evaluate prediction quality and latent representation statistics

total_mse = 0.0
total_baseline_mse = 0.0
total_cosine_similarity = 0.0
total_samples = 0

all_embeddings = []

max_batches = 50

with torch.no_grad():

    for batch_idx, (images, actions) in enumerate(test_dataloader):

        if batch_idx >= max_batches:
            break

        images = images.to(device, non_blocking=True)
        actions = actions.to(device, non_blocking=True)

        Z, Z_pred = model(images, actions)

        Z_target = Z[:, 1:, :]
        Z_previous = Z[:, :-1, :]

        mse = F.mse_loss(
            Z_pred,
            Z_target,
            reduction="mean",
        )

        baseline_mse = F.mse_loss(
            Z_previous,
            Z_target,
            reduction="mean",
        )

        cosine_similarity = F.cosine_similarity(
            Z_pred,
            Z_target,
            dim=-1,
        ).mean()

        batch_size = images.shape[0]

        total_mse += mse.item() * batch_size
        total_baseline_mse += baseline_mse.item() * batch_size
        total_cosine_similarity += cosine_similarity.item() * batch_size

        total_samples += batch_size

        all_embeddings.append(Z[:, 0, :].cpu())

# %%
# Results

avg_mse = total_mse / total_samples
avg_baseline_mse = total_baseline_mse / total_samples
avg_cosine_similarity = total_cosine_similarity / total_samples

embeddings = torch.cat(all_embeddings, dim=0)
latent_std = embeddings.std(dim=0)

print("Prediction MSE:", avg_mse)
print("Baseline MSE:", avg_baseline_mse)
print("MSE ratio:", avg_mse / avg_baseline_mse)
print("Cosine similarity:", avg_cosine_similarity)

print("Mean latent std:", latent_std.mean().item())
print("Min latent std:", latent_std.min().item())
print("Max latent std:", latent_std.max().item())

print(
    "Dimensions with std < 0.05:",
    (latent_std < 0.05).sum().item(),
    "/",
    latent_std.numel(),
)

# %%