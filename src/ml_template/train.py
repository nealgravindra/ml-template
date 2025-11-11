from __future__ import annotations

import json
import os
import random
from pathlib import Path

import click
import numpy as np
import torch
from hydra import compose, initialize_config_dir
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


def set_seed(s: int) -> None:
    random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)
    np.random.seed(s)
    # torch.backends.cudnn.deterministic = True


def make_data(n: int, d: int, noise: float, device: str) -> TensorDataset:
    X = torch.randn(n, d, device=device)
    w = torch.randn(d, 1, device=device)
    y = X @ w + noise * torch.randn(n, 1, device=device)
    return TensorDataset(X, y)


def train(
    epochs: int,
    batch_size: int,
    lr: float,
    l1: float,
    l2: float,
    n: int,
    d: int,
    noise: float,
    seed: int,
    device: str,
) -> None:
    set_seed(seed)
    device = "cuda" if (device == "auto" and torch.cuda.is_available()) else ("cpu" if device == "auto" else device)
    ds = make_data(n, d, noise, device)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True)
    model = nn.Linear(d, 1, bias=False).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    mse = nn.MSELoss()

    def enet(mse_val: torch.Tensor) -> torch.Tensor:
        l1_term = torch.stack([p.abs().sum() for p in model.parameters()]).sum()
        l2_term = torch.stack([(p**2).sum() for p in model.parameters()]).sum()
        return mse_val + l1 * l1_term + l2 * l2_term

    model.train()
    loss = None
    for _ in range(epochs):
        for xb, yb in dl:
            loss = enet(mse(model(xb), yb))

            # step
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

    os.makedirs("artifacts", exist_ok=True)
    if loss is None:
        raise RuntimeError
    with open("artifacts/metrics.json", "w") as f:
        json.dump({"final_loss": float(loss.item()), "device": device}, f)


@click.command(help="Train the model using configuration from Hydra.")
@click.option(
    "--config", type=click.Path(exists=True), required=True, help="Path to the directory containing Hydra config files."
)
def main(config: str) -> None:
    config_dir = Path(config).parent.absolute().as_posix()
    config_name = Path(config).name
    with initialize_config_dir(config_dir=config_dir):
        cfg = compose(config_name=config_name)
        train(
            epochs=cfg.train.epochs,
            batch_size=cfg.train.batch_size,
            lr=cfg.train.lr,
            l1=cfg.train.l1,
            l2=cfg.train.l2,
            n=cfg.data.n_samples,
            d=cfg.data.n_features,
            noise=cfg.data.noise,
            seed=cfg.seed,
            device=cfg.device if "device" in cfg else "auto",
        )


if __name__ == "__main__":
    main()
