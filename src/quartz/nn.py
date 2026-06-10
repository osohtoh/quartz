import numpy as np
import torch
from pathlib import Path
from torch import nn
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader, Dataset, random_split

from quartz.data import Grid


class GridDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Dataset of normalized voxel grids and scalar pKi targets."""

    def __init__(
        self,
        grids: list[Grid] | None = None,
        props: list[str] | None = None,
        inputs: list[torch.Tensor] | None = None,
        targets: list[torch.Tensor] | None = None,
        channel_mean: torch.Tensor | None = None,
        channel_std: torch.Tensor | None = None,
        split_indices: dict[str, list[int]] | None = None,
    ):
        self.props = list(props or [])
        self.split_indices = split_indices

        if inputs is not None and targets is not None:
            self.inputs = [x.clone() for x in inputs]
            self.targets = [y.clone() for y in targets]
            if channel_mean is None or channel_std is None:
                stats_tensor = torch.stack(self.inputs, dim=0)
                self.channel_mean = stats_tensor.mean(dim=(0, 2, 3, 4)).view(
                    -1, 1, 1, 1
                )
                self.channel_std = (
                    stats_tensor.std(dim=(0, 2, 3, 4), unbiased=False)
                    .view(-1, 1, 1, 1)
                    .clamp_min(1e-6)
                )
            else:
                self.channel_mean = torch.as_tensor(channel_mean).clone()
                self.channel_std = torch.as_tensor(channel_std).clone()
            return

        if grids is None:
            raise ValueError(
                "Either grids or precomputed inputs/targets must be provided"
            )
        if not self.props:
            raise ValueError("Props must contain at least one channel name")

        inputs_list: list[torch.Tensor] = []
        targets_list: list[float] = []

        for grid in grids:
            pki = float(grid.pki)
            if np.isnan(pki):
                continue

            channels = [
                np.asarray(grid.npGrid[prop], dtype=np.float32) for prop in self.props
            ]
            stacked = np.stack(channels, axis=0)
            inputs_list.append(torch.from_numpy(stacked))
            targets_list.append(pki)

        if not inputs_list:
            raise ValueError("No grids with valid pKi values were found")

        stats_tensor = torch.stack(inputs_list, dim=0)
        self.channel_mean = stats_tensor.mean(dim=(0, 2, 3, 4)).view(-1, 1, 1, 1)
        self.channel_std = (
            stats_tensor.std(dim=(0, 2, 3, 4), unbiased=False)
            .view(-1, 1, 1, 1)
            .clamp_min(1e-6)
        )

        self.inputs = [(x - self.channel_mean) / self.channel_std for x in inputs_list]
        self.targets = [
            torch.tensor([pki], dtype=torch.float32) for pki in targets_list
        ]

    def create_splits(
        self,
        test_split: float,
        val_split: float,
        seed: int | None = None,
    ) -> dict[str, list[int]]:
        n_samples = len(self)
        if n_samples < 3:
            raise ValueError("Need at least three samples to create dataset splits")

        test_size = max(1, int(round(n_samples * test_split)))
        val_size = max(1, int(round(n_samples * val_split)))
        train_size = n_samples - test_size - val_size

        if train_size < 1:
            train_size = 1
            remaining = n_samples - train_size
            if test_size >= val_size:
                test_size = remaining - 1
                val_size = 1
            else:
                val_size = remaining - 1
                test_size = 1

        generator = torch.Generator().manual_seed(seed) if seed is not None else None
        indices = torch.randperm(n_samples, generator=generator).tolist()

        train_indices = indices[:train_size]
        test_indices = indices[train_size : train_size + test_size]
        val_indices = indices[train_size + test_size :]

        self.split_indices = {
            "train": train_indices,
            "test": test_indices,
            "val": val_indices,
        }
        return self.split_indices

    def save(self, path: str | Path) -> None:
        payload = {
            "inputs": self.inputs,
            "targets": self.targets,
            "channel_mean": self.channel_mean,
            "channel_std": self.channel_std,
            "props": self.props,
            "split_indices": self.split_indices,
        }
        torch.save(payload, path)

    def __len__(self) -> int:
        return len(self.inputs)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.inputs[idx], self.targets[idx]


class SavedGridDataset(GridDataset):
    def __init__(self, source: str | Path | dict[str, object]):
        if isinstance(source, (str, Path)):
            payload = torch.load(source, weights_only=False)
        else:
            payload = source

        inputs = payload.get("inputs")
        targets = payload.get("targets")
        if not isinstance(inputs, list) or not isinstance(targets, list):
            raise ValueError("Saved dataset payload must contain inputs and targets")

        props_obj = payload.get("props")
        props = list(props_obj) if isinstance(props_obj, list) else []

        channel_mean = payload.get("channel_mean")
        if channel_mean is not None and not isinstance(channel_mean, torch.Tensor):
            raise ValueError("Saved dataset payload has an invalid channel_mean")

        channel_std = payload.get("channel_std")
        if channel_std is not None and not isinstance(channel_std, torch.Tensor):
            raise ValueError("Saved dataset payload has an invalid channel_std")

        split_indices = payload.get("split_indices")
        if split_indices is not None and not isinstance(split_indices, dict):
            raise ValueError("Saved dataset payload has an invalid split_indices")

        super().__init__(
            grids=None,
            props=props,
            inputs=inputs,
            targets=targets,
            channel_mean=channel_mean,
            channel_std=channel_std,
            split_indices=split_indices,
        )


class PocketNet(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv3d(in_channels, 16, kernel_size=5, padding=1, stride=2),
            nn.BatchNorm3d(16),
            nn.ReLU(),
            nn.Conv3d(16, 32, kernel_size=5, padding=1, stride=2),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.Conv3d(32, 32, kernel_size=5, padding=1, stride=2),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Conv3d(32, 64, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.Conv3d(64, 64, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Conv3d(64, 128, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm3d(128),
            nn.ReLU(),
            nn.Conv3d(128, 128, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm3d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool3d(1),
        )
        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.regressor(x)


def _resolve_device(device: str) -> str:
    if device.startswith("cuda") and not torch.cuda.is_available():
        return "cpu"
    return device


def train_model(
    grids: list[Grid] | GridDataset | SavedGridDataset | None = None,
    props: list[str] | None = None,
    epochs: int = 100,
    batch_size: int = 16,
    lr: float = 1e-3,
    test_split: float = 0.2,
    val_split: float = 0.1,
    device: str = "cpu",
    seed: int | None = None,
    checkpoint_every: int = 50,
    checkpoint_dir: str = "checkpoints",
    log_every_epoch: bool = True,
    data: list[Grid] | GridDataset | SavedGridDataset | None = None,
) -> tuple[PocketNet, dict[str, object]]:
    """Train PocketNet and return the model with loss history and normalization stats."""

    source = data if data is not None else grids
    if source is None:
        raise ValueError("Provide grids or data to train_model")

    if isinstance(source, Dataset):
        dataset = source
    else:
        if not props:
            raise ValueError("Props must be provided when training from raw grids")
        dataset = GridDataset(source, props)
    n_samples = len(dataset)
    if n_samples < 3:
        raise ValueError(
            "Need at least three valid samples to create train/test/val splits"
        )

    test_size = max(1, int(round(n_samples * test_split)))
    val_size = max(1, int(round(n_samples * val_split)))
    train_size = n_samples - val_size - test_size

    if train_size < 1:
        train_size = 1
        remaining = n_samples - train_size
        if test_size >= val_size:
            test_size = remaining - 1
            val_size = 1
        else:
            val_size = remaining - 1
            test_size = 1

    # seed for reproducability
    if seed is not None:
        torch.manual_seed(seed)
        split_generator = torch.Generator().manual_seed(seed)
        loader_generator = torch.Generator().manual_seed(seed)
    else:
        split_generator = None
        loader_generator = None

    train_ds, test_ds, val_ds = random_split(
        dataset,
        [train_size, test_size, val_size],
        generator=split_generator,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        generator=loader_generator,
    )
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    print(
        f"Split sizes | train={len(train_ds)} | val={len(val_ds)} | test={len(test_ds)}"
    )

    resolved_device = _resolve_device(device)
    if props:
        in_channels = len(props)
    else:
        in_channels = dataset[0][0].shape[0]
    model = PocketNet(in_channels=in_channels).to(resolved_device)

    # MSELoss for regression tasks
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=120, T_mult=2, eta_min=1e-5)

    train_loss_history: list[float] = []
    test_loss_history: list[float] = []
    history: dict[str, object] = {
        "train_loss": train_loss_history,
        "test_loss": test_loss_history,
        "channel_mean": dataset.channel_mean.clone(),
        "channel_std": dataset.channel_std.clone(),
        "split_sizes": {
            "train": len(train_ds),
            "test": len(test_ds),
            "val": len(val_ds),
        },
    }
    checkpoint_path = Path(checkpoint_dir)
    checkpoint_path.mkdir(parents=True, exist_ok=True)

    for epoch in range(epochs):
        if log_every_epoch:
            print(f"Starting epoch {epoch + 1}/{epochs}")

        model.train()
        train_loss_sum = 0.0

        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(resolved_device)
            y_batch = y_batch.to(resolved_device)

            optimizer.zero_grad()
            pred = model(x_batch)
            loss = loss_fn(pred, y_batch)
            loss.backward()
            optimizer.step()

            train_loss_sum += float(loss.item()) * x_batch.size(0)

        if (epoch + 1) % checkpoint_every == 0:
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "epoch": epoch,
                    "props": props,
                    "channel_mean": dataset.channel_mean,
                    "channel_std": dataset.channel_std,
                },
                checkpoint_path / f"pocketnet_epoch_{epoch + 1}.pt",
            )

        train_epoch_loss = train_loss_sum / len(train_ds)

        model.eval()
        test_loss_sum = 0.0
        with torch.no_grad():
            for x_batch, y_batch in test_loader:
                x_batch = x_batch.to(resolved_device)
                y_batch = y_batch.to(resolved_device)
                pred = model(x_batch)
                loss = loss_fn(pred, y_batch)
                test_loss_sum += float(loss.item()) * x_batch.size(0)

        test_epoch_loss = test_loss_sum / len(val_ds)

        train_loss_history.append(train_epoch_loss)
        test_loss_history.append(test_epoch_loss)

        if log_every_epoch:
            print(
                f"Epoch {epoch + 1}/{epochs} | "
                f"train_loss={train_epoch_loss:.6f} | "
                f"test_loss={test_epoch_loss:.6f}"
            )

        scheduler.step()

    model.eval()
    val_loss_sum = 0.0
    with torch.no_grad():
        for x_batch, y_batch in val_loader:
            x_batch = x_batch.to(resolved_device)
            y_batch = y_batch.to(resolved_device)
            pred = model(x_batch)
            loss = loss_fn(pred, y_batch)
            val_loss_sum += float(loss.item()) * x_batch.size(0)

    val_loss = val_loss_sum / len(val_ds)
    history["val_loss"] = val_loss
    print(f"Final validation loss: {val_loss:.6f}")

    return model, history


def predict(
    model: PocketNet,
    grid: Grid,
    props: list[str],
    channel_mean: torch.Tensor,
    channel_std: torch.Tensor,
    device: str = "cpu",
) -> float:
    """Predict pKi for one grid using precomputed normalization statistics."""

    resolved_device = _resolve_device(device)
    model = model.to(resolved_device)
    model.eval()

    channels = [np.asarray(grid.npGrid[prop], dtype=np.float32) for prop in props]
    x = torch.from_numpy(np.stack(channels, axis=0)).unsqueeze(0).to(resolved_device)

    mean = torch.as_tensor(
        channel_mean, dtype=torch.float32, device=resolved_device
    ).view(-1, 1, 1, 1)
    std = (
        torch.as_tensor(channel_std, dtype=torch.float32, device=resolved_device)
        .view(-1, 1, 1, 1)
        .clamp_min(1e-6)
    )

    x = (x - mean) / std

    with torch.no_grad():
        pred = model(x)

    return float(pred.squeeze().detach().cpu().item())
