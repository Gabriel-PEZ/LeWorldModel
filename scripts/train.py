# %%

from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from leworldmodel.data.pusht import PushTDataset
from leworldmodel.loss import SIGReg
from leworldmodel.model import LeWorldModel

# %%

project_root = next(
    p for p in [Path.cwd(), *Path.cwd().parents]
    if (p / "pyproject.toml").exists()
)
print(project_root)

device = "cuda" if torch.cuda.is_available() else "cpu"

torch.manual_seed(3072)

B = 16
T = 4
K = 5

epochs = 10
learning_rate = 5e-5
weight_decay = 1e-3
lamb_reg = 0.09
warmup_epochs = 1

print(device)

# %%
"""
each episode = a full trajectory 
(o_0, a_0, o_1, a_1, ..., o_T)
frame skip K = 5, (o_0, o_5, o_10, ...)
             A_0 = (a_0, a_1, a_2, a_3, a_4)
For a given episode, we build : 
S_0 = (o_0, o_5, o_10, o_15) C_0 = (A_0, A_1, A_2), A_0 = (a_0, ..., a_4), A_1 = (a_5, ..., a_9), A_2 = (a_10, ..., a_14)
S_1 = (o_5, o_10, o_15, o_20) (sample_stride = 5, if sample_stride = 1 then S_1 = (o_1, o_6, o_11, o_16))
S_2 = (o_10, o_15, o_20, o_25)
# ...
"""
dataset = PushTDataset(
    path=project_root / "data" / "pusht" / "pusht_expert_train.h5",
    sequence_length=T,
    frame_skip=K,
    max_episodes=1000,
    sample_stride=5,
)

# Split by episode, not by sequence

with h5py.File(dataset.path, "r") as f:
    episode_idx = np.asarray(f["episode_idx"][:]).flatten()

print(episode_idx)

# %%

sequence_episodes = episode_idx[dataset.starts]
unique_episodes = np.unique(sequence_episodes)

rng = np.random.default_rng(3072)
rng.shuffle(unique_episodes)

n_train = int(0.9 * len(unique_episodes))

train_episodes = unique_episodes[:n_train]
val_episodes = unique_episodes[n_train:]

train_indices = np.flatnonzero(
    np.isin(sequence_episodes, train_episodes)
)
val_indices = np.flatnonzero(
    np.isin(sequence_episodes, val_episodes)
)

train_dataset = Subset(dataset, train_indices.tolist())
val_dataset = Subset(dataset, val_indices.tolist())

assert len(np.intersect1d(train_episodes, val_episodes)) == 0

print("Total unique episodes in HDF5:", len(np.unique(episode_idx)))
print("Selected episodes:", len(unique_episodes))
print("Total sequences:", len(dataset))

# %%

"""
Select B sequences. 
For example one sequence from episode 123, one from 61, another from 1042, etcetc...
and each sequence = a window of 4 observations with frame skip K (=5) + the associated actions
WE NEVER MIXED TWO EPISODES
"""
generator = torch.Generator().manual_seed(3072)

dataloader = DataLoader(
    train_dataset,
    batch_size=B,
    shuffle=True,
    drop_last=True,
    num_workers=0,
    pin_memory=(device == "cuda"),
    generator=generator,
)

val_dataloader = DataLoader(
    val_dataset,
    batch_size=B,
    shuffle=False,
    drop_last=True,
    num_workers=0,
    pin_memory=(device == "cuda"),
)

print("Number of train episodes:", len(train_episodes))
print("Number of validation episodes:", len(val_episodes))
print("Number of train sequences:", len(train_dataset))
print("Number of validation sequences:", len(val_dataset))
print("Batches per epoch:", len(dataloader))

# %%

images, actions = next(iter(dataloader))

print("Images:", images.shape)
print("Actions:", actions.shape)

# %%

params_encoder = {
    "num_layers":12,
    "patch_size":14,
    "image_size":224,
    "num_heads":3,
    "embedding_dim":192,
    "QK_dim":64,
    "V_dim":64,
    "hidden_dim_mlp":768,
    "hidden_dim_proj":2048
}

params_predictor = {
    "num_layers":6,
    "num_heads":16,
    "embedding_dim":192,
    "QK_dim":64,
    "V_dim":64,
    "hidden_dim_mlp":2048,
    "p_dropout":0.1,
    "frame_skip":5,
    "action_dim":2,
    "hidden_dim_proj":2048,
    "N":3
}

model = LeWorldModel(
    params_encoder=params_encoder,
    params_predictor=params_predictor
).to(device)

# %%

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=learning_rate,
    weight_decay=weight_decay,
)

warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
    optimizer, # lr of the optimizer 
    start_factor=0.1, # start at 0.1 * lr
    end_factor=1.0, # stop at lr
    total_iters=warmup_epochs, # scheduler.step only after each epoch so it goes from 0.1 lr during the first epoch to lr for the second and then it starts decreasing
)

cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=epochs - warmup_epochs,
)

# SequentialLR pour enchaîner plusieurs schedulers sur un même opti
# milestones = étapes auxquelles on passe d'un scheduler au suivant
scheduler = torch.optim.lr_scheduler.SequentialLR(
    optimizer,
    schedulers=[warmup_scheduler, cosine_scheduler],
    milestones=[warmup_epochs],
)

# %%

checkpoint_dir = project_root / "checkpoints" / "run_02"
checkpoint_dir.mkdir(parents=True, exist_ok=True)

best_val_loss = float("inf")

history = {
    "train_loss": [],
    "val_loss": [],
    "train_mse": [],
    "val_mse": [],
    "train_sigreg": [],
    "val_sigreg": [],
}

# %%

for epoch in range(epochs):

    model.train()

    total_loss = 0.0
    total_mse = 0.0
    total_sigreg = 0.0

    for batch_idx, (images, actions) in enumerate(dataloader):

        images = images.to(device, non_blocking=True)
        actions = actions.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        Z, Z_pred = model(images, actions)

        loss_mse = F.mse_loss(Z_pred, Z[:, 1:, :])
        loss_sigreg = SIGReg(
            Z, M=1024, a=0.0, b=3.0, J=17, lamb=1.0
        )

        loss = loss_mse + lamb_reg * loss_sigreg

        loss.backward()

        # g = g * min(1, max_norm/||g||_2)
        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0,
        )

        optimizer.step()

        total_loss += loss.item()
        total_mse += loss_mse.item()
        total_sigreg += loss_sigreg.item()

        if (batch_idx + 1) % 100 == 0:
            print(
                f"Epoch {epoch + 1}/{epochs} | "
                f"Batch {batch_idx + 1}/{len(dataloader)} | "
                f"Loss {loss.item():.4f}",
                flush=True,
            )

    avg_loss = total_loss / len(dataloader)
    avg_mse = total_mse / len(dataloader)
    avg_sigreg = total_sigreg / len(dataloader)

    # Validation

    model.eval()

    val_total_loss = 0.0
    val_total_mse = 0.0
    val_total_sigreg = 0.0

    with torch.no_grad():

        for images, actions in val_dataloader:

            images = images.to(device, non_blocking=True)
            actions = actions.to(device, non_blocking=True)

            Z, Z_pred = model(images, actions)

            loss_mse = F.mse_loss(Z_pred, Z[:, 1:, :])
            loss_sigreg = SIGReg(
                Z, M=1024, a=0.0, b=3.0, J=17, lamb=1.0
            )

            loss = loss_mse + lamb_reg * loss_sigreg

            val_total_loss += loss.item()
            val_total_mse += loss_mse.item()
            val_total_sigreg += loss_sigreg.item()

    avg_val_loss = val_total_loss / len(val_dataloader)
    avg_val_mse = val_total_mse / len(val_dataloader)
    avg_val_sigreg = val_total_sigreg / len(val_dataloader)

    scheduler.step()

    history["train_loss"].append(avg_loss)
    history["val_loss"].append(avg_val_loss)
    history["train_mse"].append(avg_mse)
    history["val_mse"].append(avg_val_mse)
    history["train_sigreg"].append(avg_sigreg)
    history["val_sigreg"].append(avg_val_sigreg)

    print(
        f"Epoch {epoch + 1}/{epochs} | "
        f"Train loss: {avg_loss:.4f} | "
        f"Val loss: {avg_val_loss:.4f} | "
        f"Train MSE: {avg_mse:.4f} | "
        f"Val MSE: {avg_val_mse:.4f}",
        flush=True,
    )

    checkpoint = {
        "epoch": epoch + 1,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "train_loss": avg_loss,
        "val_loss": avg_val_loss,
        "params_encoder": params_encoder,
        "params_predictor": params_predictor,
        "history": history,
        "train_episodes": train_episodes.tolist(),
        "val_episodes": val_episodes.tolist(),
    }

    torch.save(
        checkpoint,
        checkpoint_dir / "last.pt",
    )

    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss

        torch.save(
            checkpoint,
            checkpoint_dir / "best.pt",
        )

        print(
            f"New best checkpoint | Val loss: {best_val_loss:.4f}",
            flush=True,
        )

# %%